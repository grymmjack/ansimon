#!/usr/bin/env python3
"""Compare the *style* of two sets of ANSI art, numerically.

    python3 tools/style-report.py REFERENCE_DIR [CANDIDATE_DIR]

With one argument it profiles a corpus. With two it scores the candidate
against the reference — which is how you tell whether a style LoRA actually
learned anything, instead of squinting at two pictures and hoping.

What it measures, and why these three
-------------------------------------
**Colour usage** (foreground on inked cells, background on all cells).
Palette habit is the most legible part of a text-mode artist's style, because
there are only 16 choices and everyone spends them differently. Backgrounds
matter separately: "draws on black" vs "fills the canvas" is a whole aesthetic.

**Glyph usage.** Which of the block/shade characters get reached for, and how
often. A smooth `░▒▓█` ramp reads very differently from hard `█`/`▀` edges.

**Tonal transition.** For every horizontally adjacent inked pair, how far apart
their colours are in luminance. An artist who models volume moves in small
steps; a quantizer approximating a photo jumps. This is the one that separates
"drawn" from "converted", and it is the hardest for a style LoRA to fix.

The score is 100 minus the total variation distance between the distributions,
so 100 is identical and 0 is disjoint. Treat it as a dial, not a grade — it
tells you which way a change moved things, which is what you need mid-training.
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_HERE, "custom_nodes"),
           os.path.expanduser("~/ComfyUI/custom_nodes")):
    if os.path.isdir(os.path.join(_p, "ansi_quantize")):
        sys.path.insert(0, _p)
        break
from ansi_quantize import ansi as A, xbin as X          # noqa: E402
from ansi_quantize.cp437 import CP437                   # noqa: E402
from ansi_quantize.palette import ANSI16                # noqa: E402

NAMES = ["black", "blue", "green", "cyan", "red", "magenta", "brown", "grey",
         "dkgrey", "br.blue", "br.green", "br.cyan", "br.red", "br.mag",
         "yellow", "white"]
# Rec.601 luma of each palette entry, normalised — used for the tonal metric.
LUMA = np.array([0.299 * r + 0.587 * g + 0.114 * b for r, g, b in ANSI16]) / 255.0
TRACKED = [0x20, 0xB0, 0xB1, 0xB2, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xFE]


def load(path):
    raw = open(path, "rb").read()
    sauce, body = A.read_sauce(raw)
    if raw[:5] == X.MAGIC:
        p = X.parse_xbin(body)
        return p["ch"], p["fg"], p["bg"]
    return A.parse_ans(raw)


def profile(files):
    fg = np.zeros(16); bg = np.zeros(16)
    gl = np.zeros(257)                      # 256 glyphs + one "other" bin
    steps = []
    n = 0
    for f in files:
        try:
            ch, f_, b_ = load(f)
        except Exception:
            continue
        ink = (ch != 0x20) & (ch != 0x00)
        if not ink.any():
            continue
        n += 1
        np.add.at(fg, f_[ink].ravel(), 1)
        np.add.at(bg, b_.ravel(), 1)
        np.add.at(gl, ch.ravel(), 1)

        # Tonal transition: |Δluma| between horizontally adjacent inked cells.
        lum = LUMA[f_]
        both = ink[:, :-1] & ink[:, 1:]
        if both.any():
            steps.append(np.abs(lum[:, :-1] - lum[:, 1:])[both])

    norm = lambda a: a / max(a.sum(), 1)
    steps = np.concatenate(steps) if steps else np.zeros(1)
    # Bucket the transitions coarsely; exact values are noise, the shape is not.
    hist, _ = np.histogram(steps, bins=[0, .05, .15, .3, .5, 1.01])
    return {"n": n, "fg": norm(fg), "bg": norm(bg), "glyph": norm(gl),
            "tone": norm(hist.astype(float)), "mean_step": float(steps.mean())}


def tvd(p, q):
    """Total variation distance -> a 0-100 similarity score."""
    return 100.0 * (1.0 - 0.5 * np.abs(p - q).sum())


def bar(v, width=22):
    return "█" * int(round(v * width)) + "·" * (width - int(round(v * width)))


def show(ref, cand=None, ref_name="reference", cand_name="candidate"):
    if cand is None:
        print(f"\n  {ref_name}: {ref['n']} pieces\n")
        print(f"  {'colour':<10}{'fg':>7}  {'':<24}{'bg':>7}")
        for i in np.argsort(-ref["fg"])[:10]:
            print(f"  {NAMES[i]:<10}{100*ref['fg'][i]:6.1f}%  "
                  f"{bar(ref['fg'][i]/max(ref['fg'].max(),1e-9))}{100*ref['bg'][i]:6.1f}%")
        print(f"\n  {'glyph':<10}{'share':>7}")
        for c in TRACKED:
            if ref["glyph"][c] > 0.001:
                print(f"  {CP437[c]!r:<10}{100*ref['glyph'][c]:6.1f}%  "
                      f"{bar(ref['glyph'][c]/max(ref['glyph'].max(),1e-9))}")
        print(f"\n  mean tonal step between adjacent cells: {ref['mean_step']:.3f}")
        print()
        return

    print(f"\n  {ref_name}: {ref['n']} pieces   vs   {cand_name}: {cand['n']} pieces\n")
    scores = {k: tvd(ref[k], cand[k]) for k in ("fg", "bg", "glyph", "tone")}
    print(f"  {'colour':<10}{ref_name[:8]:>9}{cand_name[:9]:>10}   delta")
    rows = sorted(range(16), key=lambda i: -max(ref["fg"][i], cand["fg"][i]))
    for i in rows[:10]:
        d = 100 * (cand["fg"][i] - ref["fg"][i])
        flag = "  <<" if abs(d) >= 5 else ""
        print(f"  {NAMES[i]:<10}{100*ref['fg'][i]:8.1f}%{100*cand['fg'][i]:9.1f}%"
              f"{d:+8.1f}{flag}")
    print(f"\n  {'glyph':<10}{ref_name[:8]:>9}{cand_name[:9]:>10}   delta")
    for c in TRACKED:
        if max(ref["glyph"][c], cand["glyph"][c]) < 0.005:
            continue
        d = 100 * (cand["glyph"][c] - ref["glyph"][c])
        print(f"  {CP437[c]!r:<10}{100*ref['glyph'][c]:8.1f}%{100*cand['glyph'][c]:9.1f}%{d:+8.1f}")
    print(f"\n  black background   {100*ref['bg'][0]:7.0f}%{100*cand['bg'][0]:9.0f}%")
    print(f"  mean tonal step    {ref['mean_step']:7.3f} {cand['mean_step']:8.3f}")
    print(f"\n  {'SCORES (100 = identical)':<28}")
    for k, label in (("fg", "foreground palette"), ("bg", "background palette"),
                     ("glyph", "glyph vocabulary"), ("tone", "tonal transitions")):
        print(f"    {label:<22}{scores[k]:6.1f}  {bar(scores[k]/100)}")
    print(f"    {'OVERALL':<22}{np.mean(list(scores.values())):6.1f}\n")


def files_in(p):
    p = os.path.abspath(os.path.expanduser(p))
    if os.path.isfile(p):
        return [p]
    return sorted(glob.glob(os.path.join(p, "*.ans")) +
                  glob.glob(os.path.join(p, "*.xb")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("candidate", nargs="?")
    a = ap.parse_args()

    ref_files = files_in(a.reference)
    if not ref_files:
        sys.exit(f"no .ans/.xb in {a.reference}")
    ref = profile(ref_files)
    if not a.candidate:
        show(ref, ref_name=os.path.basename(a.reference.rstrip("/")) or "corpus")
        return
    cand_files = files_in(a.candidate)
    if not cand_files:
        sys.exit(f"no .ans/.xb in {a.candidate}")
    show(ref, profile(cand_files),
         os.path.basename(a.reference.rstrip("/")) or "ref",
         os.path.basename(a.candidate.rstrip("/")) or "cand")


if __name__ == "__main__":
    main()
