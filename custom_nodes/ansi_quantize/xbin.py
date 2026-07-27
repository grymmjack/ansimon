"""XBin (`.xb`) — the art-scene container format, and the better one for us.

`.ANS` is a *stream*: escape codes and characters replayed into a terminal.
That makes width an act of faith — the file never states how wide it is, so a
132-column piece opens as 80 columns and shreds unless a SAUCE record happens
to be attached and the viewer happens to read it.

XBin is a *file*: width and height are in the header, so any canvas size is
first-class. That alone makes it the right target for game placeholder art at
arbitrary cell dimensions. Two more things it does that .ANS cannot:

  * **Embeds the font.** ansimon *generates* its block and shade glyphs,
    because the console fonts on a typical Linux box are missing exactly those
    characters. Embedding the 4 KB bitmap means the file renders identically in
    PabloDraw, Moebius, or anything else — it is not relying on the viewer
    having the right CP437 font.
  * **Embeds the palette.** The 16 colours ship with the art, so the PNG
    ansimon renders and the .xb a viewer opens agree exactly. No "why is this
    blue different" between machines.

Format (all little-endian):

    0   4   "XBIN"
    4   1   0x1A (EOF marker)
    5   2   width in characters
    7   2   height in characters
    9   1   font height in pixels (1-32)
    10  1   flags
    11  48  palette, if flags & 0x01   (16 x RGB, each 0-63)
    ..  N   font, if flags & 0x02      (fontheight x 256 bytes)
    ..  ..  image data

    flags: 0x01 palette  0x02 font  0x04 compressed
           0x08 non-blink (iCE)  0x10 512-char font

Attribute byte is the VGA one: `bg<<4 | fg`, where bit 7 is blink unless the
non-blink flag is set, in which case it is the background intensity bit.
"""
import numpy as np

from .cp437 import CELL_H, CELL_W, glyph_bitmaps

MAGIC = b"XBIN\x1a"

FLAG_PALETTE = 0x01
FLAG_FONT = 0x02
FLAG_COMPRESS = 0x04
FLAG_NONBLINK = 0x08
FLAG_512CHARS = 0x10

# Compression run types, in the high 2 bits of a control byte.
RUN_NONE = 0b00        # `count` char/attr pairs follow verbatim
RUN_CHAR = 0b01        # one char byte, then `count` attr bytes
RUN_ATTR = 0b10        # one attr byte, then `count` char bytes
RUN_BOTH = 0b11        # one char + one attr, repeated `count` times
MAX_RUN = 64           # the count field is 6 bits, stored as count-1


def pack_font(chars=None):
    """Our 8x16 glyph table -> XBin font block (256 glyphs x 16 bytes, MSB first).

    `chars` is ignored; XBin fonts are always the full 256 (or 512) glyph page,
    so we ship the whole table even when the art only used nine characters.
    """
    bits = glyph_bitmaps()                       # (256, 16, 8) bool
    return np.packbits(bits, axis=-1).astype(np.uint8).tobytes()


def pack_palette(pal):
    """16 RGB tuples (0-255) -> XBin palette block (0-63 per channel).

    XBin stores VGA DAC values, which are 6-bit. Dividing by 4 is the standard
    conversion and is what every scene tool expects; going through round() here
    rather than a bit-shift keeps 0xFF mapping to 63 instead of 62.
    """
    out = bytearray()
    for r, g, b in list(pal)[:16]:
        for v in (r, g, b):
            out.append(min(63, int(round(float(v) / 255.0 * 63.0))))
    while len(out) < 48:
        out.append(0)
    return bytes(out)


def _attrs(fg, bg):
    """(fg, bg) arrays -> VGA attribute bytes."""
    return ((np.asarray(bg, np.uint8) & 0x0F) << 4) | (np.asarray(fg, np.uint8) & 0x0F)


def compress_rows(ch, attr):
    """RLE-compress the cell stream, one row at a time.

    Runs deliberately never cross a row boundary. The spec permits it, but
    several readers in the wild assume line-aligned runs, and the size win from
    crossing is negligible on art that is mostly flat horizontal bands anyway.
    """
    out = bytearray()
    rows, cols = ch.shape
    for y in range(rows):
        rc, ra = ch[y], attr[y]
        x = 0
        while x < cols:
            c0, a0 = int(rc[x]), int(ra[x])

            # How far does each kind of run reach from here?
            both = 1
            while (both < MAX_RUN and x + both < cols
                   and rc[x + both] == c0 and ra[x + both] == a0):
                both += 1
            same_c = 1
            while (same_c < MAX_RUN and x + same_c < cols and rc[x + same_c] == c0):
                same_c += 1
            same_a = 1
            while (same_a < MAX_RUN and x + same_a < cols and ra[x + same_a] == a0):
                same_a += 1

            # Pick the run that encodes the most cells per byte spent.
            # both: 3 bytes for `both` cells.  char/attr: 2+n for n cells.
            # none: 1+2n for n cells. Compare bytes-per-cell and take the best.
            cand = [
                (RUN_BOTH, both, 3.0 / both),
                (RUN_CHAR, same_c, (2.0 + same_c) / same_c),
                (RUN_ATTR, same_a, (2.0 + same_a) / same_a),
            ]
            kind, n, cost = min(cand, key=lambda t: t[2])

            if cost >= 2.0:
                # No run beats storing raw pairs; gather a literal stretch and
                # stop it as soon as a genuine run (>= 2 identical cells) shows
                # up, so we don't swallow the start of a compressible region.
                n = 1
                while n < MAX_RUN and x + n < cols:
                    if (x + n + 1 < cols and rc[x + n] == rc[x + n + 1]
                            and ra[x + n] == ra[x + n + 1]):
                        break
                    n += 1
                out.append((RUN_NONE << 6) | (n - 1))
                for i in range(n):
                    out += bytes((int(rc[x + i]), int(ra[x + i])))
            elif kind == RUN_BOTH:
                out.append((RUN_BOTH << 6) | (n - 1))
                out += bytes((c0, a0))
            elif kind == RUN_CHAR:
                out.append((RUN_CHAR << 6) | (n - 1))
                out.append(c0)
                out += bytes(int(v) for v in ra[x:x + n])
            else:
                out.append((RUN_ATTR << 6) | (n - 1))
                out.append(a0)
                out += bytes(int(v) for v in rc[x:x + n])
            x += n
    return bytes(out)


def to_xbin(ch, fg, bg, palette=None, ice=True, compress=True,
            embed_font=True, embed_palette=True):
    """Serialise a cell grid to XBin bytes."""
    ch = np.asarray(ch, np.uint8)
    rows, cols = ch.shape
    attr = _attrs(fg, bg)

    flags = 0
    if ice:
        flags |= FLAG_NONBLINK
    if embed_palette and palette is not None:
        flags |= FLAG_PALETTE
    if embed_font:
        flags |= FLAG_FONT
    if compress:
        flags |= FLAG_COMPRESS

    out = bytearray(MAGIC)
    out += int(cols).to_bytes(2, "little")
    out += int(rows).to_bytes(2, "little")
    out.append(CELL_H)
    out.append(flags)
    if flags & FLAG_PALETTE:
        out += pack_palette(palette)
    if flags & FLAG_FONT:
        out += pack_font()
    out += compress_rows(ch, attr) if compress else \
        bytes(v for pair in zip(ch.ravel(), attr.ravel()) for v in (int(pair[0]), int(pair[1])))
    return bytes(out)


def write_xbin(path, ch, fg, bg, palette=None, ice=True, compress=True,
               embed_font=True, embed_palette=True):
    data = to_xbin(ch, fg, bg, palette, ice, compress, embed_font, embed_palette)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


# ---------------------------------------------------------------------------
# Reader — used by the self-test to prove what we wrote round-trips.
# ---------------------------------------------------------------------------
def parse_xbin(data):
    """XBin bytes -> dict with ch/fg/bg arrays plus the embedded font/palette."""
    if data[:5] != MAGIC:
        raise ValueError("not an XBin file")
    cols = int.from_bytes(data[5:7], "little")
    rows = int.from_bytes(data[7:9], "little")
    fontheight = data[9]
    flags = data[10]
    i = 11

    palette = None
    if flags & FLAG_PALETTE:
        raw = data[i:i + 48]
        i += 48
        palette = [tuple(int(round(v / 63.0 * 255.0)) for v in raw[k:k + 3])
                   for k in range(0, 48, 3)]

    font = None
    if flags & FLAG_FONT:
        n = 512 if flags & FLAG_512CHARS else 256
        raw = data[i:i + fontheight * n]
        i += fontheight * n
        font = (np.unpackbits(np.frombuffer(raw, np.uint8))
                  .reshape(n, fontheight, 8).astype(bool))

    total = cols * rows
    ch = np.zeros(total, np.uint8)
    at = np.zeros(total, np.uint8)

    if flags & FLAG_COMPRESS:
        p, k = i, 0
        while k < total and p < len(data):
            ctrl = data[p]
            p += 1
            kind, n = ctrl >> 6, (ctrl & 0x3F) + 1
            if kind == RUN_NONE:
                for _ in range(n):
                    ch[k], at[k] = data[p], data[p + 1]
                    p += 2
                    k += 1
            elif kind == RUN_CHAR:
                c = data[p]
                p += 1
                for _ in range(n):
                    ch[k], at[k] = c, data[p]
                    p += 1
                    k += 1
            elif kind == RUN_ATTR:
                a = data[p]
                p += 1
                for _ in range(n):
                    ch[k], at[k] = data[p], a
                    p += 1
                    k += 1
            else:
                c, a = data[p], data[p + 1]
                p += 2
                for _ in range(n):
                    ch[k], at[k] = c, a
                    k += 1
    else:
        raw = data[i:i + total * 2]
        ch = np.frombuffer(raw, np.uint8)[0::2].copy()
        at = np.frombuffer(raw, np.uint8)[1::2].copy()

    return {
        "cols": cols, "rows": rows, "fontheight": fontheight, "flags": flags,
        "ice": bool(flags & FLAG_NONBLINK),
        "palette": palette, "font": font,
        "ch": ch[:total].reshape(rows, cols),
        "fg": (at[:total] & 0x0F).reshape(rows, cols),
        "bg": (at[:total] >> 4).reshape(rows, cols),
    }
