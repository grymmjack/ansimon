#!/usr/bin/env python3
"""Prove ansimon's PNG and its .ANS agree, at every colour depth.

    python3 tools/verify-depth.py

The rule this enforces is the one that cost this project three separate bugs:
**validate against pixelview, never against yourself.** A round trip through
our own writer and our own reader proves the two are self-consistent, which is
exactly what a wrong glyph table, a wrong colour order and a double linefeed
all survived. So the reference here is an independent renderer.

For each depth we quantize a test image, render it ourselves, write the .ANS,
render THAT with pixelview, and require the two PNGs to be bit-identical. If
they differ the .ans is lying about what ansimon drew, whatever the preview
looks like.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_HERE, "custom_nodes"),
           os.path.expanduser("~/ComfyUI/custom_nodes")):
    if os.path.isdir(os.path.join(_p, "ansi_quantize")):
        sys.path.insert(0, _p)
        break

from ansi_quantize import ansi as A                              # noqa: E402
from ansi_quantize.cp437 import charset_indices                  # noqa: E402
from ansi_quantize.nodes import (match_cells, match_cells_deep,  # noqa: E402
                                 render_cells, render_cells_rgb, rgb_fallback)
from ansi_quantize.palette import (ANSI16, parse_palette_full,          # noqa: E402
                                   xterm256_palette)

PIXELVIEW = os.environ.get("PIXELVIEW") or os.path.expanduser(
    "~/git/pixel-viewer/target/release/pixelview")
COLS, ROWS, CELL_H, CELL_W = 80, 40, 16, 8


def test_image():
    """Something with gradients, flat fields and hard edges — all three matter.

    Gradients are where deep colour should pull away from 16; flat fields are
    where a wrong background shows; hard diagonals are where the glyph choice
    (not the colour) does the work.
    """
    h, w = ROWS * CELL_H, COLS * CELL_W
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3))
    img[..., 0] = x / w * 255                        # red ramp across
    img[..., 1] = y / h * 255                        # green ramp down
    img[..., 2] = ((x + y) % 96) / 96 * 255          # diagonal banding
    r = np.hypot(y - h / 2, x - w / 2)
    img[r < 120] = (255, 255, 0)                     # flat disc
    img[np.abs(y - x * 0.5) < 6] = (0, 128, 255)     # hard diagonal edge
    return np.clip(img, 0, 255)


def patches_of(arr):
    return (arr.reshape(ROWS, CELL_H, COLS, CELL_W, 3)
               .transpose(0, 2, 1, 3, 4)
               .reshape(ROWS * COLS, CELL_H * CELL_W, 3))


def run(depth, dialect, chars, arr, tmp, lock=None):
    """One (depth, dialect) case. `lock` is a palette to pin rgb colours to."""
    pal = np.asarray(ANSI16, np.float64)
    if depth == "16":
        ch, fg, bg, cnt, _ = match_cells(patches_of(arr), chars, pal, ice=True,
                                         shade_blend=False)
        rp = pal.astype(np.uint8)
    else:
        if depth == "256":
            # Pass a numpy array, because that is what the node passes. An
            # earlier version of this test handed in a plain list and so never
            # reached the `base16 or ANSI16` truth-test that raised on every
            # real 256 render.
            dp = np.asarray(xterm256_palette(np.asarray(ANSI16, np.uint8)),
                            np.float64)
        elif lock is not None:
            dp = np.asarray(lock, np.float64)
        else:
            dp = None
        ch, fg, bg, cnt = match_cells_deep(patches_of(arr), chars,
                                           depth=depth, pal=dp)
        rp = None if depth == "rgb" else dp.astype(np.uint8)

    ch = ch.reshape(ROWS, COLS).astype(np.uint8)
    shape = (ROWS, COLS, 3) if depth == "rgb" else (ROWS, COLS)
    fg = fg.reshape(shape).astype(np.uint8)
    bg = bg.reshape(shape).astype(np.uint8)
    cnt = cnt.reshape(ROWS, COLS)
    bg[cnt >= CELL_H * CELL_W] = 0
    fg[cnt <= 0] = 0 if depth == "rgb" else 7

    mine = (render_cells_rgb(ch, fg, bg) if depth == "rgb"
            else render_cells(ch, fg, bg, rp))

    fb = rgb_fallback(fg, bg, ANSI16, True) if depth == "rgb" else None
    ans = os.path.join(tmp, f"d{depth}_{dialect}{'_lock' if lock else ''}.ans")
    A.write_ans(ans, ch, fg, bg, ice=True, sauce=True, title="depth test",
                author="grymmjack", depth=depth, dialect=dialect, fallback=fb)

    png = ans[:-4] + "_pv.png"
    r = subprocess.run([PIXELVIEW, "--render", ans, "-o", png],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(png):
        return None, f"pixelview failed: {r.stderr.strip()[:200]}"

    theirs = np.asarray(Image.open(png).convert("RGB"))
    if theirs.shape != mine.shape:
        return None, f"size {theirs.shape[1]}x{theirs.shape[0]} vs ours " \
                     f"{mine.shape[1]}x{mine.shape[0]}"
    diff = int((theirs != mine).any(-1).sum())
    err = float(np.abs(theirs.astype(int) - arr).mean())
    return (diff, err, os.path.getsize(ans), len(np.unique(
        theirs.reshape(-1, 3), axis=0))), None


def main():
    if not os.path.exists(PIXELVIEW):
        sys.exit(f"pixelview not found at {PIXELVIEW} — build it, or set "
                 f"$PIXELVIEW. This test is meaningless without an independent "
                 f"renderer.")
    arr = test_image()
    chars = charset_indices("blocks")
    tmp = tempfile.mkdtemp(prefix="ansimon-depth-")
    total = ROWS * COLS * CELL_H * CELL_W

    print(f"\n  {COLS}x{ROWS} cells, charset 'blocks', vs pixelview\n")
    print(f"  {'depth':<14}{'differing px':>14}{'mean err':>11}"
          f"{'colours':>10}{'.ans':>10}")
    print("  " + "-" * 59)
    # A >16-colour .GPL, locked, is the case that only depth rgb can express:
    # a .ANS carrying an artist's own palette exactly, with no dependence on the
    # viewer's colour table.
    locked = parse_palette_full("ENDESGA-64")

    bad = 0
    for depth, dialect, lock in (("16", "-", None), ("256", "-", None),
                                 ("rgb", "pablo", None), ("rgb", "xterm", None),
                                 ("rgb", "pablo", locked)):
        res, why = run(depth, dialect, chars, arr, tmp, lock)
        name = depth if dialect == "-" else f"{depth}/{dialect}"
        if lock:
            name = f"rgb/lock({len(lock)})"
        if why:
            print(f"  {name:<14}{'FAIL':>14}   {why}")
            bad += 1
            continue
        diff, err, size, ncol = res
        mark = "MATCH" if diff == 0 else f"{diff} ({diff/total:.2%})"
        if diff:
            bad += 1
        print(f"  {name:<14}{mark:>14}{err:>11.2f}{ncol:>10}{size:>10}")

    print(f"\n  {'mean err' :<14}= average per-channel distance from the source "
          f"image;\n  {'':14}  lower is a better picture, and is the whole "
          f"point of deep colour.")
    print(f"\n  files in {tmp}\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
