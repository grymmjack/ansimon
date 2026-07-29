"""The 16 ANSI colours, in ANSI attribute order.

Index == the colour number you write into an escape sequence, which is also
the order the bundled `EGA (16).GPL` in pixelmon lists them in — that is not a
coincidence, it is the CGA hardware palette both things descend from.

Extra palettes can be dropped into `gpl/` as GIMP `.GPL` files (same
convention as pixelmon's `pixelart_palette`). They must contain exactly 16
colours in ANSI order to be usable, since index 9 *means* "bright blue" to
every ANSI viewer regardless of what RGB you map it to.
"""
import os
import re

# Canonical CGA/EGA/ANSI palette. 0-7 normal, 8-15 bright.
ANSI16 = [
    (0x00, 0x00, 0x00),   # 0  black
    (0x00, 0x00, 0xAA),   # 1  blue
    (0x00, 0xAA, 0x00),   # 2  green
    (0x00, 0xAA, 0xAA),   # 3  cyan
    (0xAA, 0x00, 0x00),   # 4  red
    (0xAA, 0x00, 0xAA),   # 5  magenta
    (0xAA, 0x55, 0x00),   # 6  brown
    (0xAA, 0xAA, 0xAA),   # 7  light grey
    (0x55, 0x55, 0x55),   # 8  dark grey
    (0x55, 0x55, 0xFF),   # 9  bright blue
    (0x55, 0xFF, 0x55),   # 10 bright green
    (0x55, 0xFF, 0xFF),   # 11 bright cyan
    (0xFF, 0x55, 0x55),   # 12 bright red
    (0xFF, 0x55, 0xFF),   # 13 bright magenta
    (0xFF, 0xFF, 0x55),   # 14 bright yellow
    (0xFF, 0xFF, 0xFF),   # 15 white
]

# A few well-known re-mappings of the same 16 slots. The art is identical;
# only the rendered PNG changes, exactly like viewing a .ANS in a different
# terminal emulator.
BUILTIN = {
    "ansi": ANSI16,
    "xterm": [
        (0, 0, 0), (0, 0, 238), (0, 205, 0), (0, 205, 205),
        (205, 0, 0), (205, 0, 205), (205, 205, 0), (229, 229, 229),
        (127, 127, 127), (92, 92, 255), (0, 255, 0), (0, 255, 255),
        (255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255),
    ],
    "vga-soft": [
        (0, 0, 0), (0, 0, 168), (0, 168, 0), (0, 168, 168),
        (168, 0, 0), (168, 0, 168), (168, 87, 0), (168, 168, 168),
        (84, 84, 84), (84, 84, 255), (84, 255, 84), (84, 255, 255),
        (255, 84, 84), (255, 84, 255), (255, 255, 84), (255, 255, 255),
    ],
}

_HERE = os.path.dirname(os.path.abspath(__file__))

# Palettes are searched in the bundled gpl/ folder, then anywhere on
# $ANSIMON_PALETTES (a colon-separated list), then a couple of conventional
# spots. Drop a .GPL in any of them and it becomes selectable.
GPL_DIR = os.path.join(_HERE, "gpl")
_EXTRA = [p for p in os.environ.get("ANSIMON_PALETTES", "").split(":") if p]
GPL_DIRS = [GPL_DIR] + _EXTRA + [
    os.path.expanduser("~/git/DRAW/ASSETS/PALETTES"),
    os.path.expanduser("~/.config/ansimon/palettes"),
]


def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"bad hex color: {h!r}")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _parse_gpl(path):
    """Parse a GIMP .GPL -> list of (r,g,b). Only the first 3 ints per line."""
    colors = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith(("gimp palette", "name:", "columns:")):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                colors.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    return colors


def _load_gpl_dir(directory):
    """Load every .GPL in `directory` -> {name: [(r,g,b), ...]}.

    Palettes with FEWER than 16 entries are kept and padded by repeating the
    last colour. A 4-colour CGA set is a perfectly good target — it just means
    several ANSI indices resolve to the same RGB, which is exactly what those
    machines did. Rejecting them would throw away most of a real collection:
    of 55 palettes here, 21 have fewer than 16 colours.

    More than 16 are truncated for the 16-colour path; the full list is kept
    under `ALL_PALETTES_FULL` for the 256-colour mode.
    """
    out = {}
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if not fn.lower().endswith(".gpl"):
            continue
        name = re.sub(r"\s*\(\d+\)\s*$", "", os.path.splitext(fn)[0]).strip()
        cols = _parse_gpl(os.path.join(directory, fn))
        if cols:
            out[name] = cols
    return dict(sorted(out.items()))


def _to16(cols):
    """Any palette -> exactly 16 entries, padding short ones by repetition."""
    cols = list(cols)[:16]
    while len(cols) < 16:
        cols.append(cols[-1])
    return cols


_FOUND = {}
for _d in GPL_DIRS:
    for _k, _v in _load_gpl_dir(_d).items():
        _FOUND.setdefault(_k, _v)          # earlier directories win

# Full-precision palettes (any length) for the 256-colour path.
ALL_PALETTES_FULL = {**{k: list(v) for k, v in BUILTIN.items()}, **_FOUND}
# 16-entry versions for .ANS / XBin, which have exactly 16 attributes.
ALL_PALETTES = {**BUILTIN, **{k: _to16(v) for k, v in _FOUND.items()}}


def parse_palette(name, custom_hex=""):
    """Resolve a palette name (or 'Custom' + hex list) to 16 (r,g,b) tuples."""
    if name == "Custom":
        toks = custom_hex.replace(",", " ").split()
        cols = [hex_to_rgb(t) for t in toks]
        if len(cols) != 16:
            raise ValueError(
                f"ANSI needs exactly 16 colours in attribute order, got {len(cols)}")
        return cols
    if name not in ALL_PALETTES:
        raise ValueError(f"unknown palette {name!r} — have: {', '.join(ALL_PALETTES)}")
    return ALL_PALETTES[name]


def palette_size(name):
    """How many colours the source .GPL actually had (before padding to 16)."""
    return len(ALL_PALETTES_FULL.get(name, ()))
