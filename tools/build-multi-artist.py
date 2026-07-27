#!/usr/bin/env python3
"""Build a weighted, multi-artist LoRA dataset from extracted scene art.

    python3 tools/build-multi-artist.py ARTISTS_DIR OUT_DIR \\
        --primary grymmjack --primary-extra "/path/to/local/backup" \\
        --primary-repeats 4 --other-repeats 1

Two problems this solves that the single-folder builder cannot.

**Weighting.** Extracting a whole crew gives you breadth, but if the artist you
actually want is only 21% of the files, training flat produces a "mid-90s
scene" model rather than theirs. kohya's repeat count is per folder, so
`4_grymmjack/` beside `1_filth/` makes the primary 4x as influential per epoch
while everyone else supplies compositional variety — which is the direct fix
for a model that memorised layouts.

**Separate trigger words.** Other artists are captioned under their OWN handle,
not the primary's. Tagging someone else's work `grymmjack` would teach the
trigger to mean "any mid-90s blockart", which is the opposite of the goal.
Their pieces still teach general structure; they just don't pollute the token.

Deduplication is by parsed CELL GRID, not by file bytes: the same piece ships
in several packs with different SAUCE records and line endings, but its
characters and attributes are identical. This also merges a personal backup
with the archive copy cleanly — measured on one corpus, a 112-piece backup and
a 104-piece archive extract shared only 63, for a union of 153.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_HERE), "custom_nodes"),
           os.path.expanduser("~/ComfyUI/custom_nodes")):
    if os.path.isdir(os.path.join(_p, "ansi_quantize")):
        sys.path.insert(0, _p)
        break
import numpy as np                                       # noqa: E402
from ansi_quantize import ansi as A, xbin as X           # noqa: E402


def cellhash(path):
    """Content identity: the parsed grid, immune to SAUCE and line-ending drift."""
    try:
        raw = open(path, "rb").read()
        if raw[:5] == X.MAGIC:
            p = X.parse_xbin(A.read_sauce(raw)[1])
            ch, fg, bg = p["ch"], p["fg"], p["bg"]
        else:
            ch, fg, bg = A.parse_ans(raw)
    except Exception:
        return None
    if ch.size == 0:
        return None
    h = hashlib.sha1()
    for arr in (ch, fg, bg):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def art_files(d):
    out = []
    for e in ("*.ans", "*.xb", "*.xbin", "*.ANS"):
        out += glob.glob(os.path.join(d, e))
    return sorted(f for f in out if os.path.isfile(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artists_dir", help="output of extract-artists.py")
    ap.add_argument("out", help="dataset root to build")
    ap.add_argument("--primary", required=True)
    ap.add_argument("--primary-extra", action="append", default=[],
                    help="additional source dir(s) for the primary artist")
    ap.add_argument("--primary-repeats", type=int, default=4)
    ap.add_argument("--other-repeats", type=int, default=1)
    ap.add_argument("--upscale", type=int, default=1,
                    help="1 for SD1.5 (640x384), 2 for SDXL (1280x768)")
    ap.add_argument("--rows", type=int, default=24)
    a = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(a.artists_dir))
    out = os.path.abspath(os.path.expanduser(a.out))
    stage = os.path.join(out, "_staged")
    os.makedirs(stage, exist_ok=True)

    artists = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
    pslug = re.sub(r"[^a-z0-9]+", "-", a.primary.lower()).strip("-")

    # --- stage each artist, deduped by cell grid -------------------------
    counts, total_dupes = {}, 0
    for art in artists:
        srcs = [os.path.join(root, art)]
        if art == pslug:
            srcs += [os.path.abspath(os.path.expanduser(p)) for p in a.primary_extra]
        dst = os.path.join(stage, art)
        os.makedirs(dst, exist_ok=True)
        seen, n, dup = set(), 0, 0
        for s in srcs:
            for f in art_files(s):
                h = cellhash(f)
                if h is None:
                    continue
                if h in seen:
                    dup += 1
                    continue
                seen.add(h)
                ext = os.path.splitext(f)[1].lower() or ".ans"
                shutil.copy2(f, os.path.join(dst, f"{n:04d}_{os.path.basename(f)[:60]}{'' if os.path.basename(f).lower().endswith(ext) else ext}"))
                n += 1
        counts[art] = n
        total_dupes += dup
        print(f"  staged {art:<14} {n:>4} unique  ({dup} duplicate(s) dropped)")

    # --- render each into its own weighted folder ------------------------
    print()
    builder = os.path.join(_HERE, "build-lora-dataset.py")
    merged, made = [], {}
    for art in artists:
        if counts[art] == 0:
            continue
        reps = a.primary_repeats if art == pslug else a.other_repeats
        token = a.primary if art == pslug else art.replace("-", " ")
        r = subprocess.run([sys.executable, builder, os.path.join(stage, art), out,
                            "--token", token, "--repeats", str(reps),
                            "--upscale", str(a.upscale), "--rows", str(a.rows)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ! {art}: {r.stderr.strip()[:160]}")
            continue
        mpath = os.path.join(out, "manifest.json")
        if os.path.exists(mpath):
            recs = json.load(open(mpath, encoding="utf-8"))
            for rec in recs:
                rec["artist"] = token
                rec["repeats"] = reps
            merged += recs
            made[token] = (len(recs), reps)
        line = [l for l in r.stdout.splitlines() if "samples written" in l]
        print(f"  built  {token:<14} {line[0].split()[0] if line else '?':>4} samples  x{reps}")

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=1)

    eff = {t: n * r for t, (n, r) in made.items()}
    tot = sum(eff.values()) or 1
    print(f"\n  {len(merged)} samples total, {tot} effective per epoch\n")
    for t, e in sorted(eff.items(), key=lambda kv: -kv[1]):
        n, r = made[t]
        print(f"    {t:<14} {n:>4} x{r}  = {e:>5}  ({100*e/tot:4.1f}%)")
    print(f"\n  staging kept at {stage} (safe to delete)\n")


if __name__ == "__main__":
    main()
