#!/usr/bin/env python3
"""Acceptance gauntlet: ~100 renders across every axis, each verified.

    python3 tools/gauntlet.py --list          # show the matrix, render nothing
    python3 tools/gauntlet.py                 # run it
    python3 tools/gauntlet.py --verify-only   # re-check what's already there

Every case is checked the only way that means anything: render the `.ans` (and
`.xb`) with **pixelview** and require it to be bit-identical to the PNG ansimon
produced. A case that differs by one pixel is a failure, because the PNG is
ansimon's claim about what the art file contains.

Runtime comes almost entirely from sampling, not quantizing. ComfyUI caches by
node inputs, so cases sharing a (subject, seed, canvas, res, cell) key reuse the
sampled image and re-quantize in about a second. The cases are therefore SORTED
by that key before running — ~100 outputs cost ~30 samples. Changing depth,
palette, charset, shading or format is free; changing canvas size or cell
geometry is not, because those change the latent dimensions.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser("~/ComfyUI/.venv/bin/python")
PIXELVIEW = os.environ.get("PIXELVIEW") or os.path.expanduser(
    "~/git/pixel-viewer/target/release/pixelview")
OUT = os.path.expanduser("~/ansimon-lora/tests")

# --- the matrix -------------------------------------------------------------
# A "shot" is what costs GPU time: subject + seed + canvas + cell geometry.
# A "look" is free: how the quantizer interprets that same sampled image.
#
#   shot: (tag, subject, seed, cols, rows, res, vga50, font9)
#   look: (suffix, *flags)
SHOTS = [
    # -- classic DOS canvas ------------------------------------------------
    ("skull",     "a human skull",                                 1001, 80,  25, 1024, 0, 0),
    ("dragon",    "a fierce dragon head, scales",                   1002, 80,  25, 1024, 0, 0),
    ("knight",    "a knight in plate armour",                       1003, 80,  25, 1024, 0, 0),
    # -- default canvas ----------------------------------------------------
    ("city",      "a neon cyberpunk city street at night, rain",    1004, 80,  40, 1024, 0, 0),
    ("wizard",    "a wizard casting a spell, robes, staff",         1005, 80,  40, 1024, 0, 0),
    ("castle",    "a castle on a cliff at sunset",                  1006, 80,  40, 1024, 0, 0),
    ("forest",    "a dark forest path, fog, moonlight",             1007, 80,  40, 1024, 0, 0),
    ("mech",      "a battle mech, heavy armour, hydraulics",        1008, 80,  40, 1024, 0, 0),
    ("kraken",    "a kraken attacking a sailing ship",              1009, 80,  40, 1024, 0, 0),
    ("volcano",   "an erupting volcano, lava, ash cloud",           1010, 80,  40, 1024, 0, 0),
    # -- 8x8 cell (VGA50) --------------------------------------------------
    ("phoenix",   "a phoenix rising, flames, wings spread",         1011, 80,  50, 1024, 1, 0),
    ("station",   "a space station orbiting a gas giant",           1012, 80,  50, 1024, 1, 0),
    ("samurai",   "a samurai helmet, ornate crest",                 1013, 80,  50, 1024, 1, 0),
    # -- 9-dot cell --------------------------------------------------------
    ("border",    "an art deco ornamental border, symmetrical",     1014, 80,  25, 1024, 0, 1),
    ("circuit",   "a dense circuit board pattern, traces",          1015, 80,  40, 1024, 0, 1),
    # -- wide canvases -----------------------------------------------------
    ("waterfall", "a waterfall in a jungle canyon",                 1016, 132, 50, 1152, 0, 0),
    ("desert",    "desert ruins under two moons",                   1017, 132, 60, 1152, 0, 0),
    ("aurora",    "aurora borealis over a frozen lake",             1018, 160, 50, 1216, 0, 0),
    ("train",     "a steam locomotive at speed, smoke",             1019, 200, 60, 1280, 0, 0),
    # -- small / tile canvases --------------------------------------------
    ("potion",    "a glowing potion bottle",                        1020, 40,  20, 768,  0, 0),
    ("chest",     "a treasure chest overflowing with gold",         1021, 40,  20, 768,  0, 0),
    ("sword",     "an ornate sword, jewelled hilt",                 1022, 64,  32, 896,  0, 0),
    ("crown",     "a jewelled crown",                               1023, 64,  32, 896,  0, 0),
    ("pumpkin",   "a carved jack-o-lantern, candle inside",         1024, 48,  48, 896,  0, 0),
    ("icon",      "a simple skull icon, high contrast",             1025, 24,  12, 640,  0, 0),
    # -- remaining subjects to broaden the gamut ---------------------------
    ("reactor",   "a fusion reactor core, containment ring",        1026, 100, 30, 1024, 0, 0),
    ("lighthouse","a lighthouse in a storm, huge waves",            1027, 100, 30, 1024, 0, 0),
    ("cyborg",    "a cyborg soldier, glowing optics",               1028, 80,  40, 1024, 0, 0),
    ("mandala",   "a symmetrical geometric mandala",                1029, 80,  40, 1024, 0, 0),
    ("underwater","a sunken city, shafts of light, fish",           1030, 80,  40, 1024, 0, 0),
]

# Every shot gets `BASE` — depth 16 is the default anyone will actually use, so
# every subject and every canvas is exercised there at minimum. The deeper modes
# are added per shot below, weighted toward gradient-heavy subjects where they
# have something to prove, rather than uniformly.
BASE = [
    ("d16", "--depth", "16"),
]

EXTRA = {
    # depth 16, palette + charset + shading spread
    "skull":     [("ega-blocks", "--palette", "EGA", "--charset", "blocks"),
                  ("lock-nes", "--truecolor", "--lock-palette", "--palette", "NES")],
    "dragon":    [("c64", "--palette", "C=64", "--shading", "medium"),
                  ("lock-sega", "--truecolor", "--lock-palette", "--palette", "SEGA")],
    "knight":    [("pico8-dither", "--palette", "PICO-8", "--dither"),
                  ("xb", "--format", "xb"),
                  ("lock-e64", "--truecolor", "--lock-palette", "--palette", "ENDESGA-64")],
    "city":      [("cyan-lock", "--colors", "3,8,11,15"),
                  ("lock-neons", "--truecolor", "--lock-palette",
                   "--palette", "CYBERPUNK-NEONS"),
                  ("xterm-dialect", "--truecolor", "--rgb-dialect", "xterm")],
    "wizard":    [("db16-full", "--palette", "DAWNBRINGER-16", "--charset", "full"),
                  ("lock-db32", "--truecolor", "--lock-palette",
                   "--palette", "DAWNBRINGER-32")],
    "castle":    [("zx-geometric", "--palette", "ZXSPECTRUM", "--charset", "geometric"),
                  ("both", "--format", "both"),
                  ("lock-quake", "--truecolor", "--lock-palette", "--palette", "QUAKE")],
    "forest":    [("noice", "--no-ice", "--shading", "medium"),
                  ("lock-vga", "--truecolor", "--lock-palette", "--palette", "VGA")],
    "mech":      [("structure", "--charset", "structure"),
                  ("msx", "--palette", "MSX", "--shading", "full"),
                  ("lock-atari", "--truecolor", "--lock-palette",
                   "--palette", "ATARI-8BIT")],
    "kraken":    [("cga0", "--palette", "CGA0-HIGH", "--charset", "blocks"),
                  ("lock-e32", "--truecolor", "--lock-palette",
                   "--palette", "ENDESGA-32")],
    "volcano":   [("hallow", "--palette", "HALLOWPUMPKIN", "--dither"),
                  ("lock-6bit", "--truecolor", "--lock-palette", "--palette", "6BIT")],
    "phoenix":   [("blocks", "--charset", "blocks"),
                  ("lock-ansi32", "--truecolor", "--lock-palette", "--palette", "ANSI32")],
    "station":   [("xterm-pal", "--palette", "xterm", "--shading", "medium"),
                  ("lock-amstrad", "--truecolor", "--lock-palette",
                   "--palette", "AMSTRADCPC")],
    "samurai":   [("black-bg", "--black-bg"),
                  ("lock-e36", "--truecolor", "--lock-palette", "--palette", "ENDESGA-36")],
    "border":    [("blocks-9px", "--charset", "blocks"),
                  ("lock-teletext", "--truecolor", "--lock-palette",
                   "--palette", "TELETEXT")],
    "circuit":   [("geometric", "--charset", "geometric"),
                  ("lock-shovel", "--truecolor", "--lock-palette",
                   "--palette", "SHOVEL-KNIGHT-NES")],
    "waterfall": [("gameboy", "--palette", "GAMEBOY", "--charset", "blocks"),
                  ("lock-vines", "--truecolor", "--lock-palette",
                   "--palette", "VINES-FLEXIBLE-LINEAR-RAMPS")],
    "desert":    [("db32-lock", "--truecolor", "--lock-palette",
                   "--palette", "PINEAPPLE-32")],
    "aurora":    [("lock-ink", "--truecolor", "--lock-palette", "--palette", "INK")],
    "train":     [("lock-atari2600", "--truecolor", "--lock-palette",
                   "--palette", "ATARI2600")],
    "potion":    [("intellivision", "--palette", "INTELLIVISION"),
                  ("xb", "--format", "xb")],
    "chest":     [("apple2", "--palette", "APPLE2-LORES", "--shading", "medium")],
    "sword":     [("e16-both", "--palette", "ENDESGA-16", "--format", "both"),
                  ("lock-cga32", "--truecolor", "--lock-palette", "--palette", "CGA32")],
    "crown":     [("colodore", "--palette", "COLODORE", "--charset", "blocks")],
    "pumpkin":   [("ascii", "--charset", "ascii"),
                  ("lock-bloodmoon", "--truecolor", "--lock-palette",
                   "--palette", "BLOODMOON21")],
    "icon":      [("blocks", "--charset", "blocks"),
                  ("xb", "--format", "xb")],
    "reactor":   [("winpal", "--palette", "MS-WINDOWS", "--dither"),
                  ("lock-funky", "--truecolor", "--lock-palette",
                   "--palette", "FUNKYFUTURE")],
    "lighthouse":[("secam", "--palette", "SECAM", "--charset", "blocks")],
    "cyborg":    [("full-medium", "--charset", "full", "--shading", "medium"),
                  ("lock-synthe", "--truecolor", "--lock-palette",
                   "--palette", "SYNTHEWAVE-CITY")],
    "mandala":   [("bbc", "--palette", "BBCMICRO", "--charset", "geometric"),
                  ("lock-jungle", "--truecolor", "--lock-palette", "--palette", "JUNGLE-8")],
    "underwater":[("teal", "--colors", "3,11,1,9,15"),
                  ("lock-vivid", "--truecolor", "--lock-palette",
                   "--palette", "VIVIDMEMORY")],
}

# Deep colour, added where it has something to prove: gradients, glows, skies,
# reflections. Deliberately NOT uniform — a flat high-contrast icon at 24x12
# looks the same at 16 colours as at 24-bit, so spending a case on it teaches
# nothing.
for _tag in ("city", "forest", "volcano", "aurora", "underwater", "waterfall",
             "station", "phoenix", "reactor", "lighthouse", "desert", "cyborg"):
    EXTRA.setdefault(_tag, []).append(("d256", "--depth", "256"))
for _tag in ("city", "volcano", "aurora", "underwater", "waterfall", "desert",
             "train", "cyborg", "kraken", "castle"):
    EXTRA.setdefault(_tag, []).append(("drgb", "--truecolor"))


def build():
    """-> list of case dicts, sorted so cache-sharing cases run together."""
    cases = []
    for tag, subj, seed, cols, rows, res, vga50, font9 in SHOTS:
        for suffix, *flags in BASE + EXTRA.get(tag, []):
            cases.append({
                "tag": tag, "subject": subj, "seed": seed,
                "cols": cols, "rows": rows, "res": res,
                "vga50": bool(vga50), "font9": bool(font9),
                "suffix": suffix, "flags": list(flags),
            })
    # Sampling is the only expensive part; keep identical samples adjacent.
    cases.sort(key=lambda c: (c["cols"], c["rows"], c["res"], c["vga50"],
                              c["font9"], c["tag"], c["suffix"]))
    for i, c in enumerate(cases, 1):
        c["name"] = f"{i:03d}_{c['tag']}_{c['suffix']}"
    return cases


def sampler_key(c):
    return (c["subject"], c["seed"], c["cols"], c["rows"], c["res"],
            c["vga50"], c["font9"])


def ans_is_palette_faithful(c):
    """Can this case's `.ans` reproduce ansimon's own colours in a viewer?

    Only if the file actually carries the colours. It does when:

      * the palette is the standard one a viewer already uses, so indices agree
        by luck rather than by transport; or
      * the depth is `rgb`, which writes literal RGB per cell.

    At depth 16 or 256 with a custom palette the `.ans` holds bare indices and
    the viewer supplies its own RGB, so ansimon's PNG and any correct viewer
    MUST differ. That is the format's limit, not a defect — it is why XBin
    exists — so those cases are verified through their `.xb` instead, and the
    harness forces one to be written.
    """
    if "--truecolor" in c["flags"] or "rgb" in c["flags"]:
        return True
    pal = None
    if "--palette" in c["flags"]:
        pal = c["flags"][c["flags"].index("--palette") + 1]
    return pal in (None, "ansi", "EGA")


def render(c, extra_args=()):
    cmd = [PY, "-u", os.path.join(HERE, "ansimon.py"), c["subject"],
           "--lora", "ansi-art-xl.safetensors", "--lora-strength", "0.9",
           "--res", str(c["res"]), "--cols", str(c["cols"]),
           "--rows", str(c["rows"]), "--seed", str(c["seed"]),
           "--steps", "30", "--no-open", "--name", c["name"],
           "--output-to", OUT]
    if c["vga50"]:
        cmd.append("--vga50")
    if c["font9"]:
        cmd.append("--font-9px")
    cmd += c["flags"] + list(extra_args)
    if not ans_is_palette_faithful(c) and "--format" not in cmd:
        # Make sure a .xb exists, because it is the only file in this case that
        # can carry the palette and therefore the only one worth diffing.
        cmd += ["--format", "both"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    ok = "✅" in r.stdout
    err = ""
    if not ok:
        tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
        err = tail[-1][:180] if tail else "no output"
    return ok, err


def verify(c):
    """Render every art file with pixelview and diff against our PNG."""
    pngs = glob.glob(os.path.join(OUT, f"{c['name']}_*_ansi_*.png"))
    if not pngs:
        return None, "no PNG produced"
    mine = np.asarray(Image.open(pngs[0]).convert("RGB"))

    arts = sorted(glob.glob(os.path.join(OUT, f"{c['name']}_*.ans"))
                  + glob.glob(os.path.join(OUT, f"{c['name']}_*.xb")))
    if not arts:
        return None, "no .ans/.xb produced"

    faithful = ans_is_palette_faithful(c)
    if not faithful and not any(a.endswith(".xb") for a in arts):
        return None, "custom palette needs a .xb to be verifiable"

    results = []
    for art in arts:
        pv = art + ".pv.png"
        cmd = [PIXELVIEW, "--render", art, "-o", pv]
        if c["font9"]:
            cmd.append("--font-9px")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(pv):
            results.append((os.path.splitext(art)[1], None,
                            f"pixelview failed: {r.stderr.strip()[:120]}"))
            continue
        theirs = np.asarray(Image.open(pv).convert("RGB"))
        os.remove(pv)
        ext = os.path.splitext(art)[1]

        # A `.ans` cannot state its height — it is a stream, and a viewer picks
        # a screen. pixelview uses the classic 25-row minimum, so a canvas
        # shorter than that comes back padded. That is correct behaviour, not a
        # mismatch, so require the art region to match exactly AND the padding
        # to be blank. Anything else (including a canvas TALLER than ours) is a
        # real failure. `.xb` states its height in the header and gets no
        # latitude at all.
        if theirs.shape != mine.shape:
            taller = (ext == ".ans" and theirs.shape[0] > mine.shape[0]
                      and theirs.shape[1] == mine.shape[1])
            if not taller:
                results.append((ext, None,
                                f"{theirs.shape[1]}x{theirs.shape[0]} vs ours "
                                f"{mine.shape[1]}x{mine.shape[0]}"))
                continue
            pad = theirs[mine.shape[0]:]
            if pad.any():
                results.append((ext, None, "padding rows are not blank"))
                continue
            theirs = theirs[:mine.shape[0]]

        diff = int((theirs != mine).any(-1).sum())
        if ext == ".ans" and not faithful:
            # The RGB is not ours to dictate here, but the LAYOUT still is. An
            # earlier version of this check passed unconditionally and thereby
            # masked a real bug: `--charset full` was emitting 0x1B as a glyph,
            # corrupting the stream, on a case whose palette made it exempt.
            # So compare ink coverage — which pixels are background — because
            # stream corruption moves ink even when it cannot move colour.
            ink_m = (mine != mine[0, 0]).any(-1)
            ink_t = (theirs != theirs[0, 0]).any(-1)
            off = int((ink_m != ink_t).sum())
            results.append((ext + "*", off,
                            "" if off == 0 else "ink layout differs"))
            continue
        results.append((ext, diff, ""))

    ncol = len(np.unique(mine.reshape(-1, 3), axis=0))
    return {"results": results, "colours": ncol,
            "px": mine.shape[1] * mine.shape[0],
            "bytes": sum(os.path.getsize(a) for a in arts)}, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", action="store_true",
                    help="don't wipe the output dir first")
    a = ap.parse_args()

    cases = build()
    if a.limit:
        cases = cases[:a.limit]
    keys = {sampler_key(c) for c in cases}

    if a.list:
        for c in cases:
            geo = f"{c['cols']}x{c['rows']}"
            cell = "8x8" if c["vga50"] else ("9x16" if c["font9"] else "8x16")
            print(f"  {c['name']:<34} {geo:>8} {cell:>5}  {' '.join(c['flags'])}")
        print(f"\n  {len(cases)} cases, {len(keys)} distinct samples")
        return

    if not a.verify_only and not a.keep:
        for f in glob.glob(os.path.join(OUT, "*")):
            os.remove(f)
    os.makedirs(OUT, exist_ok=True)

    print(f"\n  gauntlet: {len(cases)} cases, {len(keys)} distinct samples "
          f"-> {OUT}\n")
    t0 = time.time()
    failed, mismatch, done = [], [], 0

    for i, c in enumerate(cases, 1):
        if not a.verify_only:
            ok, err = render(c)
            if not ok:
                failed.append((c["name"], err))
                print(f"  {i:>3}/{len(cases)} ❌ {c['name']:<34} {err}")
                continue
        info, why = verify(c)
        if info is None:
            mismatch.append((c["name"], why))
            print(f"  {i:>3}/{len(cases)} ❌ {c['name']:<34} {why}")
            continue
        bad = [f"{ext}:{d if d is not None else w}"
               for ext, d, w in info["results"] if d != 0]
        done += 1
        if bad:
            mismatch.append((c["name"], "; ".join(bad)))
        mark = "✅" if not bad else "⚠️ "
        exts = ",".join(e.lstrip(".") for e, _, _ in info["results"])
        print(f"  {i:>3}/{len(cases)} {mark} {c['name']:<34} "
              f"{info['colours']:>5} col  {exts:<7} "
              f"{info['bytes']/1024:>7.1f} KB"
              f"{'  ' + '; '.join(bad) if bad else ''}")

    el = time.time() - t0
    print(f"\n  {'-'*70}")
    print(f"  rendered {done}/{len(cases)} in {el/60:.1f} min "
          f"({el/max(done,1):.1f}s each, {len(keys)} samples)")
    if failed:
        print(f"\n  {len(failed)} FAILED TO RENDER:")
        for n, e in failed:
            print(f"    {n:<34} {e}")
    if mismatch:
        print(f"\n  {len(mismatch)} DID NOT MATCH PIXELVIEW:")
        for n, e in mismatch:
            print(f"    {n:<34} {e}")
    if not failed and not mismatch:
        print(f"\n  ✅ all {done} cases pixel-identical to pixelview\n")
    sys.exit(1 if (failed or mismatch) else 0)


if __name__ == "__main__":
    main()
