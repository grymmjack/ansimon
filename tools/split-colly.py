#!/usr/bin/env python3
"""Split a logo "colly" into its individual logos.

    python3 tools/split-colly.py RENDER.png OUT_DIR [--preview sheet.png]
    python3 tools/split-colly.py RENDERS_DIR OUT_DIR --min-gap 2

Scene groups capped how many pieces one artist could contribute to a pack, so
artists bundled many small logos into one "collection" file. As training data a
colly is close to useless — it teaches "a column of unrelated logos" — but each
logo inside it is a perfectly good sample.

They separate cleanly because the format forces it: a colly is a vertical stack
with blank gutters between entries, so a run of entirely-empty character rows is
a reliable boundary. Working in CELL rows (16 px) rather than pixel rows keeps
the cut aligned to the character grid, which is where the art actually lives.

Each logo is then tight-cropped to its own ink and padded back out to whole
cells, so nothing is sliced mid-character.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

CELL_H = 16


def background(arr):
    """The canvas colour: the most common pixel. Usually black, not always."""
    flat = arr.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    return cols[counts.argmax()].astype(int)


def cell_rows_blank(arr, cell_h=CELL_H, tol=12):
    """-> bool array, one entry per character row: is it entirely background?"""
    h = arr.shape[0] // cell_h
    if h == 0:
        return np.zeros(0, bool)
    band = arr[:h * cell_h].reshape(h, cell_h, arr.shape[1], 3).astype(int)
    return (np.abs(band - background(arr)).sum(-1) <= tol).all(axis=(1, 2))


def segments(blank, min_gap, min_rows):
    """Runs of non-blank cell rows, separated by >= min_gap blank rows."""
    out, start, gap = [], None, 0
    for i, b in enumerate(blank):
        if not b:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap:
                    if i - gap + 1 - start >= min_rows:
                        out.append((start, i - gap + 1))
                    start, gap = None, 0
    if start is not None and len(blank) - start >= min_rows:
        out.append((start, len(blank)))
    return out


def split_one(path, outdir, a):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    bg = background(arr)
    blank = cell_rows_blank(arr)
    segs = segments(blank, a.min_gap, a.min_rows)
    stem = os.path.splitext(os.path.basename(path))[0]
    made = []
    for n, (r0, r1) in enumerate(segs):
        tile = arr[r0 * CELL_H:r1 * CELL_H]
        # tight-crop horizontally to this logo's own ink, then round out to cells
        ink = (np.abs(tile.astype(int) - bg).sum(-1) > 12)
        if not ink.any():
            continue
        cols = np.where(ink.any(0))[0]
        x0 = (cols[0] // 8) * 8
        x1 = min(tile.shape[1], ((cols[-1] // 8) + 1) * 8)
        tile = tile[:, x0:x1]
        if tile.shape[0] < a.min_rows * CELL_H or tile.shape[1] < 8 * 4:
            continue
        out = os.path.join(outdir, f"{stem}__{n:02d}.png")
        Image.fromarray(tile).save(out)
        made.append({"file": os.path.basename(out), "source": os.path.basename(path),
                     "cell_rows": [int(r0), int(r1)],
                     "size": [int(tile.shape[1]), int(tile.shape[0])]})
    return segs, made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="a rendered .png, or a folder of them")
    ap.add_argument("out")
    ap.add_argument("--min-gap", type=int, default=1,
                    help="blank CELL rows that count as a separator. default 1")
    ap.add_argument("--min-rows", type=int, default=3,
                    help="ignore segments shorter than this many cell rows")
    ap.add_argument("--preview", default=None,
                    help="write a contact sheet of the pieces found")
    a = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(a.src))
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out, exist_ok=True)
    files = ([src] if os.path.isfile(src)
             else sorted(glob.glob(os.path.join(src, "*.png"))))
    if not files:
        sys.exit(f"no .png at {src}")

    manifest, tally = [], []
    for f in files:
        segs, made = split_one(f, out, a)
        manifest += made
        tally.append((os.path.basename(f), len(segs), len(made)))

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    multi = [t for t in tally if t[2] > 1]
    print(f"\n  {len(files)} file(s) -> {len(manifest)} pieces")
    print(f"  {len(multi)} looked like collections (>1 piece):")
    for name, s, m in sorted(multi, key=lambda t: -t[2])[:12]:
        print(f"    {m:>3} pieces   {name}")

    if a.preview and manifest:
        cols = 6
        thumbs = [Image.open(os.path.join(out, m["file"])) for m in manifest[:60]]
        tw = max(t.width for t in thumbs)
        th = max(t.height for t in thumbs)
        sheet = Image.new("RGB", (cols * (tw + 6) + 6,
                                  ((len(thumbs) + cols - 1) // cols) * (th + 6) + 6),
                          (24, 24, 24))
        for i, t in enumerate(thumbs):
            sheet.paste(t, (6 + (i % cols) * (tw + 6), 6 + (i // cols) * (th + 6)))
        sheet.save(a.preview)
        print(f"  preview: {a.preview}")
    print()


if __name__ == "__main__":
    main()
