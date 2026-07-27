#!/usr/bin/env python3
"""Write captions from the captioning page back into a LoRA dataset.

    python3 tools/apply-captions.py DATASET_DIR captions.json [--dry-run]

kohya reads the caption for `foo.png` from `foo.txt` beside it, so this just
overwrites those. The originals are kept as `foo.txt.auto` the first time, so
the machine-generated captions the dataset builder produced are never lost —
useful for comparing a hand-captioned run against the automatic one.

Any image without an entry in the JSON keeps its existing caption, so you can
caption in several sittings and apply as you go.
"""
import argparse
import glob
import json
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("captions", help="captions.json exported from the page")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(a.dataset))
    caps = json.load(open(os.path.expanduser(a.captions), encoding="utf-8"))
    imgs = {os.path.basename(p): p for p in
            glob.glob(os.path.join(root, "*", "*.png")) + glob.glob(os.path.join(root, "*.png"))}
    if not imgs:
        sys.exit(f"no .png under {root}")

    written = skipped = missing = 0
    for name, cap in caps.items():
        p = imgs.get(name)
        if not p:
            missing += 1
            continue
        txt = os.path.splitext(p)[0] + ".txt"
        if not a.dry_run:
            if os.path.exists(txt) and not os.path.exists(txt + ".auto"):
                shutil.copy2(txt, txt + ".auto")     # keep the machine version once
            with open(txt, "w", encoding="utf-8") as f:
                f.write(cap.strip() + "\n")
        written += 1
    skipped = len(imgs) - written

    print(f"\n  {'would write' if a.dry_run else 'wrote'} {written} caption(s)")
    if skipped:
        print(f"  {skipped} image(s) left with their existing caption")
    if missing:
        print(f"  {missing} JSON entr(ies) had no matching image")
    if written:
        ex = next(iter(caps.values()))
        print(f"\n  example: {ex[:100]}")
    print()


if __name__ == "__main__":
    main()
