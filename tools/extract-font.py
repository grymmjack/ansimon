#!/usr/bin/env python3
"""Extract a CP437 font from a reference renderer, by rendering and reading back.

    python3 tools/extract-font.py                    # 8x16 via pixelview
    python3 tools/extract-font.py --height 8
    python3 tools/extract-font.py --from-png chart.png --height 8

Why not just generate the glyphs
--------------------------------
An earlier version of this project generated the block characters
mathematically, on the reasoning that a half block "is the top 8 rows of the
cell". That is arithmetic, not the ROM: the real VGA upper half block covers
rows **0-6** and the lower half rows **7-15**. 111 of 256 glyphs disagreed with
the actual font, and the resulting one-pixel error sat in nearly every
half-block cell of every image.

So the font is taken from a renderer we trust rather than derived. The method
is self-validating: build a 16x16 chart containing every CP437 code, render it,
and read each cell back. If ansimon's glyphs come from the same font the
reference renderer uses, its PNG and the .ans it writes cannot disagree.

XBin is used as the carrier rather than .ANS because it stores raw char/attr
pairs — codes 0x0A, 0x0D, 0x1A and 0x1B are just characters there, where in a
.ANS stream they would be newline, EOF and escape.
"""
import argparse
import os
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
from ansi_quantize import xbin as X                    # noqa: E402

PIXELVIEW = os.environ.get("PIXELVIEW") or os.path.expanduser(
    "~/git/pixel-viewer/target/release/pixelview")
CELL_W = 8


def build_chart(path, height):
    """A 16x16 grid of every CP437 code, white on black, as XBin."""
    ch = np.arange(256, dtype=np.uint8).reshape(16, 16)
    fg = np.full((16, 16), 15, np.uint8)
    bg = np.zeros((16, 16), np.uint8)
    data = X.to_xbin(ch, fg, bg, palette=None, ice=True, compress=False,
                     embed_font=False, embed_palette=False)
    # Patch the fontsize byte so the renderer picks the cell height we want.
    data = bytearray(data)
    data[9] = height
    open(path, "wb").write(bytes(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=int, default=16, choices=[8, 14, 16])
    ap.add_argument("--from-png", default=None,
                    help="skip rendering; read an existing 16x16 CP437 chart")
    ap.add_argument("--out", default=None)
    ap.add_argument("--renderer", default=PIXELVIEW)
    a = ap.parse_args()

    h = a.height
    out = a.out or os.path.join(_HERE, "custom_nodes", "ansi_quantize",
                                f"vga{CELL_W}x{h}.bin")

    if a.from_png:
        png = os.path.expanduser(a.from_png)
    else:
        if not os.path.exists(a.renderer):
            sys.exit(f"renderer not found at {a.renderer}\n"
                     f"  build it, or pass --from-png with a chart you rendered "
                     f"another way")
        import tempfile
        td = tempfile.mkdtemp()
        chart = os.path.join(td, "chart.xb")
        png = os.path.join(td, "chart.png")
        build_chart(chart, h)
        r = subprocess.run([a.renderer, "--render", chart, "-o", png],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(png):
            sys.exit(f"render failed: {r.stderr.strip()[:300]}")

    im = Image.open(png).convert("RGB")
    want = (16 * CELL_W, 16 * h)
    if im.size != want:
        sys.exit(f"chart is {im.size}, expected {want} for a {CELL_W}x{h} cell.\n"
                 f"  The renderer probably does not support that cell height.")

    lit = np.asarray(im).sum(-1) > 128            # white glyph on black
    font = np.zeros((256, h, CELL_W), bool)
    for i in range(256):
        r, c = divmod(i, 16)
        font[i] = lit[r*h:(r+1)*h, c*CELL_W:(c+1)*CELL_W]

    np.packbits(font, axis=-1).astype(np.uint8).tofile(out)
    blank = [i for i in range(256) if not font[i].any()]
    print(f"\n  {CELL_W}x{h} font -> {out}  ({os.path.getsize(out)} bytes)")
    print(f"  blank glyphs: {len(blank)} ({', '.join(hex(b) for b in blank[:6])}"
          f"{'...' if len(blank) > 6 else ''})")
    ink = font.reshape(256, -1).mean(1)
    print(f"  mean ink {ink.mean():.2f}, fullest glyph "
          f"0x{int(ink.argmax()):02X} at {ink.max():.2f}\n")


if __name__ == "__main__":
    main()
