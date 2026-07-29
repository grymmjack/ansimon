"""CP437 8x16 glyph bitmaps — the shared source of truth for ansimon.

Everything downstream needs the same 256 glyph shapes:

  * the QUANTIZER needs them as coverage masks, to decide which character best
    approximates a cell of the source image;
  * the RENDERER needs them as bitmaps, to draw the final PNG.

Using one table for both is what keeps the PNG an honest picture of the `.ANS`
— render and quantize can never disagree about what a character looks like.

Where the shapes come from
--------------------------
`vga8x16.bin` — a real VGA font, extracted from the reference renderer by
rendering a full CP437 chart and reading the cells back (tools/extract-font.py).
It is the authority; the PSF / TTF / procedural paths below are fallbacks for a
machine without it.

This used to generate the geometric glyphs procedurally, on the reasoning that
a half block "IS the top 8 rows of the cell" and so geometry beat sampling a
font. That is arithmetic, not the ROM. The real VGA upper half block covers
rows **0-6** and the lower half rows **7-15** — seven and nine, not eight and
eight. That one-pixel error was in nearly every half-block cell, and 111 of 256
glyphs disagreed with the font in total.

The lesson generalises: the PNG being a faithful picture of the .ANS is only
meaningful against an *external* reference. Checking ansimon's renderer against
ansimon's parser proves they share assumptions, not that either is right.

Those geometric glyphs do matter — measured on 112 hand-drawn scene pieces
(179,000 non-blank cells), the five shade/half-block characters alone are 45.9%
of non-blank cells and 64.0% with the full block. Which is exactly why getting
their shape wrong was so costly.
"""
import gzip
import os
import struct

import numpy as np

CELL_W, CELL_H = 8, 16          # the default cell; 8x8 (VGA50) is also supported

# Cell height is a runtime property, not a constant. A VGA 80x25 screen uses an
# 8x16 cell; 80x50 uses 8x8 — same pixels, twice the rows. Everything that
# slices an image into cells must ask, not assume.
SUPPORTED_HEIGHTS = (16, 8)

# ---------------------------------------------------------------------------
# CP437 -> Unicode. Index = the byte you'd write into a .ANS file.
# Positions 0x00 and 0xFF are blank in practice; 0x20 is a real space.
# ---------------------------------------------------------------------------
CP437 = (
    "\x00☺☻♥♦♣♠•"
    "◘○◙♂♀♪♫☼"
    "►◄↕‼¶§▬↨"
    "↑↓→←∟↔▲▼"
    " !\"#$%&'()*+,-./0123456789:;<=>?"
    "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_"
    "`abcdefghijklmnopqrstuvwxyz{|}~⌂"
    "Çüéâäàåç"
    "êëèïîìÄÅ"
    "ÉæÆôöòûù"
    "ÿÖÜ¢£¥₧ƒ"
    "áíóúñÑªº"
    "¿⌐¬½¼¡«»"
    "░▒▓│┤╡╢╖"
    "╕╣║╗╝╜╛┐"
    "└┴┬├─┼╞╟"
    "╚╔╩╦╠═╬╧"
    "╨╤╥╙╘╒╓╫"
    "╪┘┌█▄▌▐▀"
    "αßΓπΣσµτ"
    "ΦΘΩδ∞φε∩"
    "≡±≥≤⌠⌡÷≈"
    "°∙·√ⁿ²■ "
)
assert len(CP437) == 256, f"CP437 table is {len(CP437)} chars, expected 256"

UNICODE_TO_CP437 = {c: i for i, c in enumerate(CP437) if i not in (0, 0xFF)}

# ---------------------------------------------------------------------------
# Named subsets. `--charset` on the CLI picks one of these; the quantizer is
# only ever allowed to emit characters from the active subset.
# ---------------------------------------------------------------------------
BLANK = 0x20
FULL_BLOCK = 0xDB
UPPER_HALF = 0xDF
LOWER_HALF = 0xDC
LEFT_HALF = 0xDD
RIGHT_HALF = 0xDE
SHADES = (0xB0, 0xB1, 0xB2)                      # light, medium, dark
BLOCKS = (BLANK, FULL_BLOCK, UPPER_HALF, LOWER_HALF, LEFT_HALF, RIGHT_HALF)
BOX_DRAWING = tuple(range(0xB3, 0xDB))
GEOMETRIC = tuple(sorted(set(SHADES + BLOCKS + (0xFE,))))

CHARSETS = {
    # Half-block only: the safest, most authentic blockart vocabulary.
    "halfblock": (BLANK, FULL_BLOCK, UPPER_HALF),
    # Adds the shade ramp — classic BBS gradients.
    "blocks": tuple(sorted(set(BLOCKS + SHADES))),
    # Everything geometric, incl. vertical halves and the centred square.
    "geometric": GEOMETRIC,
    # Geometric + box drawing: lets the matcher find lines and corners.
    "structure": tuple(sorted(set(GEOMETRIC + BOX_DRAWING))),
    # Printable 7-bit ASCII only — for true ASCII art, no colour blocks.
    "ascii": tuple(range(0x21, 0x7F)) + (BLANK,),
    # The whole page. Maximum fidelity, highest noise risk.
    "full": tuple(i for i in range(1, 256) if i != 0x7F),
}


# ---------------------------------------------------------------------------
# Procedural geometry. These are exact by construction — a half block is not
# an approximation of anything, it IS the top 8 rows of an 8x16 cell.
# ---------------------------------------------------------------------------
def _geometric_glyphs():
    """Return {cp437_index: (16,8) bool array} for the geometric range."""
    ys, xs = np.mgrid[0:CELL_H, 0:CELL_W]
    g = {}

    g[0x00] = np.zeros((CELL_H, CELL_W), bool)          # NUL renders as blank
    g[0x20] = np.zeros((CELL_H, CELL_W), bool)          # space
    g[0xFF] = np.zeros((CELL_H, CELL_W), bool)          # non-breaking space

    # Shade ramp. Ordered dither patterns at 25 / 50 / 75% coverage — these
    # read as intermediate tones between the fg and bg colour, which is how
    # BBS artists got gradients out of a 16-colour palette.
    g[0xB0] = (xs % 2 == 0) & (ys % 2 == 0)                        # ░ 25%
    g[0xB1] = (xs + ys) % 2 == 0                                   # ▒ 50%
    g[0xB2] = ~((xs % 2 == 1) & (ys % 2 == 1))                     # ▓ 75%

    # Blocks.
    g[0xDB] = np.ones((CELL_H, CELL_W), bool)           # █ full
    g[0xDC] = ys >= CELL_H // 2                         # ▄ lower half
    g[0xDD] = xs < CELL_W // 2                          # ▌ left half
    g[0xDE] = xs >= CELL_W // 2                         # ▐ right half
    g[0xDF] = ys < CELL_H // 2                          # ▀ upper half

    # ■ centred square, matching the VGA font's proportions.
    sq = np.zeros((CELL_H, CELL_W), bool)
    sq[4:12, 2:6] = True
    g[0xFE] = sq

    return g


# ---------------------------------------------------------------------------
# PSF console-font loading (PSF1 and PSF2), including the Unicode table that
# lets us map CP437 codepoints onto the font's own glyph ordering.
# ---------------------------------------------------------------------------
def _read_maybe_gz(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return f.read()


def _parse_unicode_table(tbl, n_glyphs, seps):
    """Walk a PSF Unicode table -> {unicode_char: glyph_index}.

    Entries are per-glyph, terminated by 0xFFFF (PSF1) / 0xFF (PSF2). A
    sequence marker (0xFFFE / 0xFE) starts a multi-codepoint ligature, which
    we skip — we only want single-character mappings.
    """
    term, seq, step = seps
    out, idx, i, cur, in_seq = {}, 0, 0, [], False
    while i + step - 1 < len(tbl) and idx < n_glyphs:
        v = int.from_bytes(tbl[i:i + step], "little")
        i += step
        if v == term:
            if not in_seq:
                for c in cur:
                    out.setdefault(c, idx)
            cur, in_seq = [], False
            idx += 1
        elif v == seq:
            in_seq = True
        elif not in_seq:
            try:
                cur.append(chr(v))
            except ValueError:
                pass
    return out


def _load_psf(path):
    """Parse a PSF1/PSF2 font -> (glyph_bitmaps, {unicode: index}) or None.

    Only 8-pixel-wide, 16-row fonts are accepted; anything else would need
    rescaling and would blur the very edges we care about.
    """
    try:
        d = _read_maybe_gz(path)
    except OSError:
        return None

    if d[:2] == b"\x36\x04":                                   # PSF1
        mode, charsize = d[2], d[3]
        if charsize != CELL_H:
            return None
        n = 512 if mode & 0x01 else 256
        body = d[4:4 + n * charsize]
        has_uni = bool(mode & 0x02)
        uni_tbl = d[4 + n * charsize:]
        seps = (0xFFFF, 0xFFFE, 2)
        width = 8
    elif d[:4] == b"\x72\xb5\x4a\x86":                          # PSF2
        (_ver, hdrsize, flags, n, charsize,
         height, width) = struct.unpack("<7I", d[4:32])
        if width != CELL_W or height != CELL_H:
            return None
        body = d[hdrsize:hdrsize + n * charsize]
        has_uni = bool(flags & 0x01)
        uni_tbl = d[hdrsize + n * charsize:]
        seps = (0xFF, 0xFE, 1)
    else:
        return None

    bytes_per_row = (width + 7) // 8
    stride = bytes_per_row * CELL_H
    glyphs = []
    for i in range(n):
        raw = body[i * stride:(i + 1) * stride]
        if len(raw) < stride:
            break
        bits = np.unpackbits(np.frombuffer(raw, np.uint8))
        glyphs.append(bits.reshape(CELL_H, bytes_per_row * 8)[:, :CELL_W].astype(bool))

    uni = _parse_unicode_table(uni_tbl, len(glyphs), seps) if has_uni else {}
    return glyphs, uni


_PSF_CANDIDATES = (
    "/usr/share/consolefonts/Uni2-VGA16.psf.gz",
    "/usr/share/consolefonts/Uni1-VGA16.psf.gz",
    "/usr/share/consolefonts/Lat15-VGA16.psf.gz",
    "/usr/share/consolefonts/Lat2-VGA16.psf.gz",
    "/usr/share/kbd/consolefonts/Uni2-VGA16.psf.gz",
    "/usr/share/kbd/consolefonts/lat2-16.psfu.gz",
)


def _psf_source():
    """First usable PSF font on this machine, or (None, {})."""
    env = os.environ.get("ANSIMON_FONT")
    for path in ([env] if env else []) + list(_PSF_CANDIDATES):
        if path and os.path.exists(path):
            got = _load_psf(path)
            if got and got[0]:
                return got
    return None, {}


def _ttf_glyph(ch, size=16):
    """Render one character to an 8x16 mask with PIL. Last-resort text source."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
        if not os.path.exists(cand):
            continue
        try:
            font = ImageFont.truetype(cand, size)
        except OSError:
            continue
        img = Image.new("L", (CELL_W, CELL_H), 0)
        ImageDraw.Draw(img).text((0, 0), ch, font=font, fill=255)
        return np.asarray(img) > 96
    return None


# ---------------------------------------------------------------------------
# The public table.
# ---------------------------------------------------------------------------
_CACHE = {}

# The authority: 256 glyphs x 16 rows, 1 bit per pixel, extracted from the
# reference renderer's own VGA font by rendering a full CP437 chart and reading
# the cells back. See tools/extract-font.py.
#
# This replaced procedurally generating the geometric glyphs. That approach was
# justified here as being "more accurate than sampling a font, because a half
# block IS the top 8 rows of the cell" — which is arithmetic, not the ROM. The
# real VGA upper half block covers rows 0-6 and the lower half rows 7-15: seven
# and nine, not eight and eight. Getting that wrong put a one-pixel error in
# nearly every half-block cell, and 111 glyphs in total disagreed with the font.
_FONT_DIR = os.path.dirname(os.path.abspath(__file__))
_FONT_BIN = os.path.join(_FONT_DIR, "vga8x16.bin")      # kept for --doctor


def font_path(height=CELL_H):
    return os.path.join(_FONT_DIR, f"vga8x{height}.bin")


def _embedded_font(height=CELL_H):
    """The extracted VGA font as (256,height,8) bool, or None if absent."""
    p = font_path(height)
    if not os.path.exists(p):
        return None
    raw = np.fromfile(p, np.uint8)
    if raw.size != 256 * height:
        return None
    return np.unpackbits(raw).reshape(256, height, 8).astype(bool)


def glyph_bitmaps(height=CELL_H):
    """(256, 16, 8) bool array — glyph_bitmaps()[c] is CP437 char `c`.

    Built once and cached. Geometric glyphs are generated and OVERRIDE any
    font-supplied version, so shades and half blocks are always pixel-exact
    regardless of which console font happened to be installed.
    """
    key = f"bitmaps{height}"
    if key in _CACHE:
        return _CACHE[key]

    font = _embedded_font(height)
    if font is not None:
        _CACHE[key] = font
        return font
    if height != CELL_H:
        raise FileNotFoundError(
            f"no {CELL_W}x{height} font at {font_path(height)}.\n"
            f"  Extract one the same way as the 8x16:  "
            f"python3 tools/extract-font.py --height {height}")

    table = np.zeros((256, CELL_H, CELL_W), bool)
    glyphs, uni = _psf_source()

    filled = set()
    if glyphs:
        for cp, ch in enumerate(CP437):
            gi = uni.get(ch)
            if gi is not None and gi < len(glyphs):
                table[cp] = glyphs[gi]
                filled.add(cp)
        # A font without a Unicode table is assumed to already be CP437-ordered.
        if not uni:
            for cp in range(min(256, len(glyphs))):
                table[cp] = glyphs[cp]
                filled.add(cp)

    # Anything the font didn't cover and that isn't geometric: try TTF.
    for cp, ch in enumerate(CP437):
        if cp in filled or cp in GEOMETRIC or ch in ("\x00", "\xa0", " "):
            continue
        m = _ttf_glyph(ch)
        if m is not None:
            table[cp] = m

    # Geometry last — always authoritative.
    for cp, mask in _geometric_glyphs().items():
        table[cp] = mask

    _CACHE["bitmaps"] = table
    return table


def cell_height():
    """Height of the font actually available, in pixels."""
    return CELL_H if _embedded_font(CELL_H) is not None else CELL_H


def coverage():
    """(256,) float array — fraction of each glyph's 128 pixels that are lit.

    This is the glyph's "ink weight", used to pick a character whose density
    matches how far a cell's colour sits between its two chosen palette
    entries. coverage()[0x20] == 0.0, coverage()[0xDB] == 1.0.
    """
    if "coverage" not in _CACHE:
        _CACHE["coverage"] = glyph_bitmaps().mean(axis=(1, 2))
    return _CACHE["coverage"]


def charset_indices(name):
    """Resolve a charset name (or a comma-list of hex codes) to a tuple of ints."""
    if name in CHARSETS:
        return CHARSETS[name]
    out = []
    for tok in name.replace(",", " ").split():
        try:
            out.append(int(tok, 16) if tok.lower().startswith("0x") else int(tok))
        except ValueError:
            raise ValueError(f"unknown charset {name!r} — try one of: "
                             f"{', '.join(CHARSETS)}")
    if not out:
        raise ValueError(f"unknown charset {name!r}")
    return tuple(sorted({c & 0xFF for c in out}))


def font_source_description():
    """Human-readable note about where the text glyphs came from (for --doctor)."""
    if _embedded_font() is not None:
        return f"bundled VGA 8x16 ({_FONT_BIN})"
    env = os.environ.get("ANSIMON_FONT")
    for path in ([env] if env else []) + list(_PSF_CANDIDATES):
        if path and os.path.exists(path):
            got = _load_psf(path)
            if got and got[0]:
                return f"PSF: {path} ({len(got[0])} glyphs)"
    if _ttf_glyph("A") is not None:
        return "TTF fallback (DejaVu/Liberation Mono @ 8x16)"
    return "procedural geometry only (no text glyphs found)"
