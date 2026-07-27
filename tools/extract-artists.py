#!/usr/bin/env python3
"""Pull one or more artists' work out of a 16colo.rs archive tree.

    python3 tools/extract-artists.py ARCHIVE_DIR OUT_DIR --artists grymmjack,tainted,jed
    python3 tools/extract-artists.py ARCHIVE_DIR OUT_DIR --artists-file artists.txt --primary grymmjack

The archive is year folders full of pack `.zip`s. Every extracted piece keeps
its attribution — artist, group, pack, year — in `manifest.json`, because SAUCE
records authorship and there is no reason to throw that away.

Matching an artist is fussier than string equality. The same person signs as
"grymmjack", "grymmjack (gj!)", "grymmjack(gj!)" and "GrymmJack" across a
decade, so names are normalised (lowercased, punctuation and parenthetical
handles stripped) before comparison. Files with no SAUCE record fall back to a
filename-prefix match, which is how scene artists have always signed their
work: `gj-borg.ans`.
"""
import argparse
import io
import json
import os
import re
import sys
import zipfile

ART_EXT = (".ans", ".xb", ".xbin", ".bin", ".asc", ".nfo", ".diz")
WANT_EXT = (".ans", ".xb", ".xbin")          # what we actually keep
SAUCE_ID = b"SAUCE"


def norm(s):
    """'grymmjack (gj!)' -> 'grymmjack'. Handles a decade of signature drift."""
    s = s.replace("\x00", " ").lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)            # drop parenthetical handles
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def read_sauce(data):
    """Return the SAUCE dict or None. Cheap: only the last 128 bytes matter."""
    cut = data.rfind(b"\x1a")
    if cut == -1 or len(data) - cut - 1 < 128 or data[cut + 1:cut + 6] != SAUCE_ID:
        return None
    r = data[cut + 1:cut + 129]
    d = lambda a, b: r[a:b].decode("cp437", "replace").replace("\x00", " ").strip()
    return {"title": d(7, 42), "author": d(42, 62), "group": d(62, 82),
            "date": d(82, 90), "datatype": r[94], "filetype": r[95],
            "cols": int.from_bytes(r[96:98], "little"),
            "rows": int.from_bytes(r[98:100], "little")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--artists", default="", help="comma-separated handles")
    ap.add_argument("--artists-file", default=None, help="one handle per line")
    ap.add_argument("--primary", default=None,
                    help="handle whose work is also copied to out/primary/")
    ap.add_argument("--prefixes", default="",
                    help="comma-separated filename prefixes to match when a file "
                         "has no SAUCE, e.g. 'gj-'")
    ap.add_argument("--min-bytes", type=int, default=400,
                    help="skip tiny files (stubs, empty stubs). default 400")
    ap.add_argument("--limit-years", default="",
                    help="e.g. 1994-2001 to narrow the scan")
    a = ap.parse_args()

    names = [n.strip() for n in a.artists.split(",") if n.strip()]
    if a.artists_file:
        names += [l.strip() for l in open(os.path.expanduser(a.artists_file))
                  if l.strip() and not l.startswith("#")]
    if not names:
        sys.exit("give --artists or --artists-file")
    wanted = {}
    for n in names:
        wanted[norm(n)] = n
    prefixes = tuple(p.strip().lower() for p in a.prefixes.split(",") if p.strip())

    root = os.path.abspath(os.path.expanduser(a.archive))
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out, exist_ok=True)

    years = sorted(d for d in os.listdir(root)
                   if d.isdigit() and os.path.isdir(os.path.join(root, d)))
    if a.limit_years:
        lo, _, hi = a.limit_years.partition("-")
        years = [y for y in years if int(lo) <= int(y) <= int(hi or lo)]

    zips = []
    for y in years:
        for f in sorted(os.listdir(os.path.join(root, y))):
            if f.lower().endswith(".zip"):
                zips.append((y, os.path.join(root, y, f)))
    print(f"  scanning {len(zips)} packs across {len(years)} years "
          f"for {len(wanted)} artist(s)\n")

    manifest, per_artist, seen = [], {}, set()
    for i, (year, zp) in enumerate(zips, 1):
        if i % 250 == 0:
            print(f"    ...{i}/{len(zips)} packs, {len(manifest)} pieces so far")
        try:
            zf = zipfile.ZipFile(zp)
        except Exception:
            continue
        pack = os.path.splitext(os.path.basename(zp))[0]
        for info in zf.infolist():
            nm = info.filename
            if info.is_dir() or not nm.lower().endswith(ART_EXT):
                continue
            if info.file_size < a.min_bytes or info.file_size > 4_000_000:
                continue
            try:
                data = zf.read(info)
            except Exception:
                continue

            s = read_sauce(data)
            key = None
            if s and s["author"]:
                key = wanted.get(norm(s["author"]))
            if key is None and prefixes and os.path.basename(nm).lower().startswith(prefixes):
                key = wanted.get(norm(a.primary or names[0]))
            if key is None:
                continue
            if not nm.lower().endswith(WANT_EXT):
                continue

            # Same piece often appears in several packs; keep the first.
            sig = (key, os.path.basename(nm).lower(), info.CRC)
            if sig in seen:
                continue
            seen.add(sig)

            slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
            dest_dir = os.path.join(out, slug)
            os.makedirs(dest_dir, exist_ok=True)
            base = f"{year}_{pack}_{os.path.basename(nm)}"
            base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
            with open(os.path.join(dest_dir, base), "wb") as fh:
                fh.write(data)

            rec = {"file": os.path.join(slug, base), "artist": key,
                   "sauce_author": s["author"] if s else "", "year": year,
                   "pack": pack, "orig": nm, "bytes": len(data),
                   "title": s["title"] if s else "",
                   "group": s["group"] if s else "",
                   "cols": s["cols"] if s else 0, "rows": s["rows"] if s else 0}
            manifest.append(rec)
            per_artist[key] = per_artist.get(key, 0) + 1

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"\n  {len(manifest)} pieces -> {out}\n")
    for k, v in sorted(per_artist.items(), key=lambda t: -t[1]):
        mark = "  <- primary" if a.primary and norm(k) == norm(a.primary) else ""
        print(f"    {v:>5}  {k}{mark}")
    print(f"\n  attribution kept in {os.path.join(out, 'manifest.json')}\n")


if __name__ == "__main__":
    main()
