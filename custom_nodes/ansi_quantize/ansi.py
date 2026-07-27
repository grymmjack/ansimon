"""Cell grid <-> `.ANS` — the actual ANSI art file format.

A "cell grid" here is three uint8 arrays of shape (rows, cols):

    ch  — CP437 character index (0-255); this byte goes straight into the file
    fg  — foreground colour, 0-15
    bg  — background colour, 0-7 (or 0-15 with iCE colours enabled)

The asymmetry between fg and bg is not an oversight, it is the hardware. VGA
text mode packs an attribute byte as `blink(1) bg(3) fg(4)` — so foreground
gets 16 colours but background only gets 8, with the top bit spent on blink.
The art scene's workaround was "iCE colours": tell the adapter to reallocate
the blink bit to intensity, and you get 16 background colours and no blinking.
`ice=True` writes that variant (and flags it in the SAUCE record so viewers
know to turn blinking off).

Colour indices are the standard ANSI/CGA order, which is exactly the order of
the bundled `EGA (16).GPL` palette:

    0 black    1 blue     2 green    3 cyan
    4 red      5 magenta  6 brown    7 light grey
    8 dark grey 9 bright blue ... 15 white
"""
import numpy as np

ESC = "\x1b"
SGR_FG = {i: 30 + i for i in range(8)}
SGR_BG = {i: 40 + i for i in range(8)}

# SAUCE — the metadata record the art scene appends to .ANS files. 128 bytes
# after an EOF (0x1A) marker. Without it, viewers have to guess the canvas
# width; with it, an 80-column piece opens as 80 columns everywhere.
SAUCE_ID = b"SAUCE"
SAUCE_VERSION = b"00"
DATATYPE_CHARACTER = 1
FILETYPE_ANSI = 1


# ---------------------------------------------------------------------------
# A tiny transport blob for the cell grid.
#
# The quantizer runs on whichever GPU the farm handed the job to, and the save
# node might want .ans, .xb, or both. Rather than have the quantizer guess and
# emit pre-serialised bytes, it emits the CELLS and lets the saver decide. One
# source of truth, and adding a new output format never touches the quantizer.
# ---------------------------------------------------------------------------
CELLS_MAGIC = b"ACEL\x01"


def pack_cells(ch, fg, bg, palette, ice):
    """Cell grid + palette -> compact bytes for the ComfyUI STRING channel."""
    ch = np.asarray(ch, np.uint8)
    rows, cols = ch.shape
    out = bytearray(CELLS_MAGIC)
    out += int(cols).to_bytes(2, "little")
    out += int(rows).to_bytes(2, "little")
    out.append(1 if ice else 0)
    for r, g, b in list(palette)[:16]:
        out += bytes((int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF))
    out += ch.tobytes()
    out += np.asarray(fg, np.uint8).tobytes()
    out += np.asarray(bg, np.uint8).tobytes()
    return bytes(out)


def unpack_cells(data):
    """Inverse of pack_cells -> (ch, fg, bg, palette, ice)."""
    if data[:5] != CELLS_MAGIC:
        raise ValueError("not an ansimon cell blob")
    cols = int.from_bytes(data[5:7], "little")
    rows = int.from_bytes(data[7:9], "little")
    ice = bool(data[9])
    pal = [tuple(data[10 + i * 3:13 + i * 3]) for i in range(16)]
    n = rows * cols
    base = 10 + 48
    a = np.frombuffer(data, np.uint8, count=n * 3, offset=base)
    return (a[:n].reshape(rows, cols).copy(),
            a[n:2 * n].reshape(rows, cols).copy(),
            a[2 * n:].reshape(rows, cols).copy(), pal, ice)


def _sgr(fg, bg, ice):
    """Build the shortest SGR sequence that selects (fg, bg) from a reset state."""
    parts = []
    if fg >= 8:
        parts.append("1")                       # bold == bright foreground
    if ice and bg >= 8:
        parts.append("5")                       # blink bit reused as bright bg
    parts.append(str(SGR_FG[fg & 7]))
    parts.append(str(SGR_BG[bg & 7]))
    return f"{ESC}[" + ";".join(parts) + "m"


def to_ans(ch, fg, bg, ice=False, width=None, trim_trailing=True):
    """Serialise a cell grid to `.ANS` bytes (CP437, no SAUCE record).

    Attribute changes are emitted only when they actually change, which is
    what keeps real ANSI files small. The catch is that turning a bright
    foreground *off* needs a full reset (`ESC[0m`) — there is no "unbold" in
    the ANSI.SYS dialect these files target — so we reset and re-issue rather
    than trying to diff attribute-by-attribute.
    """
    ch = np.asarray(ch, np.uint8)
    fg = np.asarray(fg, np.uint8)
    bg = np.asarray(bg, np.uint8)
    rows, cols = ch.shape
    width = width or cols

    out = bytearray()
    cur = None                                   # currently-active (fg, bg)

    for y in range(rows):
        row_ch, row_fg, row_bg = ch[y], fg[y], bg[y]

        # Trailing run of blank-on-black costs nothing to omit — the newline
        # gets us to the next row anyway.
        end = cols
        if trim_trailing:
            while end > 0 and row_ch[end - 1] in (0x00, 0x20) and row_bg[end - 1] == 0:
                end -= 1

        for x in range(end):
            c, f, b = int(row_ch[x]), int(row_fg[x]), int(row_bg[x])
            if c == 0x00:
                c = 0x20                          # never write a raw NUL
            # A space shows only its background, so its foreground is free —
            # inheriting the current fg avoids a pointless attribute change.
            if c == 0x20 and cur is not None:
                f = cur[0]
            if cur != (f, b):
                if cur is not None and (cur[0] >= 8 > f or (ice and cur[1] >= 8 > b)):
                    out += f"{ESC}[0m".encode("ascii")
                out += _sgr(f, b, ice).encode("ascii")
                cur = (f, b)
            out.append(c)

        if y != rows - 1:
            out += b"\r\n"

    out += f"{ESC}[0m".encode("ascii")
    return bytes(out)


def sauce_record(data_len, cols, rows, title="", author="", group="",
                 date="", ice=False, aspect_square=True,
                 datatype=DATATYPE_CHARACTER, filetype=FILETYPE_ANSI):
    """Build the 128-byte SAUCE record that describes an ANSI file.

    `date` is YYYYMMDD; callers pass one explicitly rather than reading the
    clock here so that output stays reproducible for a given seed.
    """
    def fixed(s, n):
        return s.encode("cp437", "replace")[:n].ljust(n, b" ")

    # TFlags: bit0 = non-blink (iCE), bits 1-2 letter spacing, bits 3-4 aspect.
    tflags = 0
    if ice:
        tflags |= 0x01
    tflags |= (0x01 << 1)                        # 8-pixel (no 9th-column) font
    tflags |= (0x02 if aspect_square else 0x01) << 3

    rec = bytearray()
    rec += SAUCE_ID + SAUCE_VERSION
    rec += fixed(title, 35)
    rec += fixed(author, 20)
    rec += fixed(group, 20)
    rec += fixed(date or "00000000", 8)
    rec += int(data_len).to_bytes(4, "little")
    rec += bytes([datatype, filetype])
    rec += int(cols).to_bytes(2, "little")       # TInfo1 = width
    rec += int(rows).to_bytes(2, "little")       # TInfo2 = height
    rec += (0).to_bytes(2, "little")             # TInfo3
    rec += (0).to_bytes(2, "little")             # TInfo4
    rec += bytes([0])                            # comment lines
    rec += bytes([tflags])
    rec += fixed("IBM VGA", 22)                  # TInfoS = font name
    assert len(rec) == 128, len(rec)
    return bytes(rec)


def write_ans(path, ch, fg, bg, ice=False, sauce=True, title="", author="",
              group="", date="", aspect_square=True):
    """Write a complete `.ANS` file (art + EOF marker + SAUCE)."""
    body = to_ans(ch, fg, bg, ice=ice)
    with open(path, "wb") as f:
        f.write(body)
        if sauce:
            rows, cols = np.asarray(ch).shape
            f.write(b"\x1a")
            f.write(sauce_record(len(body), cols, rows, title, author, group,
                                 date, ice, aspect_square))
    return len(body)


# ---------------------------------------------------------------------------
# Reader — used by the self-test to prove that what we wrote round-trips.
# Deliberately minimal: it understands the subset `to_ans` emits.
# ---------------------------------------------------------------------------
def parse_ans(data, cols=80, rows=None, ice=False):
    """Parse `.ANS` bytes back into (ch, fg, bg) arrays. Subset parser."""
    if data[:5] != SAUCE_ID:
        cut = data.rfind(b"\x1a")
        if cut != -1 and len(data) - cut - 1 >= 128 and data[cut + 1:cut + 6] == SAUCE_ID:
            rec = data[cut + 1:cut + 129]
            cols = int.from_bytes(rec[96:98], "little") or cols
            rows = int.from_bytes(rec[98:100], "little") or rows
            data = data[:cut]

    grid = []
    line_ch, line_fg, line_bg = [], [], []
    fg, bg, bold, blink = 7, 0, False, False
    i = 0
    while i < len(data):
        b = data[i]
        if b == 0x1B and i + 1 < len(data) and data[i + 1] == ord("["):
            j = i + 2
            while j < len(data) and not (0x40 <= data[j] <= 0x7E):
                j += 1
            if j < len(data) and data[j] == ord("m"):
                for tok in data[i + 2:j].split(b";"):
                    if not tok:
                        continue
                    v = int(tok)
                    if v == 0:
                        fg, bg, bold, blink = 7, 0, False, False
                    elif v == 1:
                        bold = True
                    elif v == 5:
                        blink = True
                    elif 30 <= v <= 37:
                        fg = v - 30
                    elif 40 <= v <= 47:
                        bg = v - 40
            i = j + 1
            continue
        if b == 0x0D:
            i += 1
            continue
        if b == 0x0A:
            grid.append((line_ch, line_fg, line_bg))
            line_ch, line_fg, line_bg = [], [], []
            i += 1
            continue
        line_ch.append(b)
        line_fg.append(fg + (8 if bold else 0))
        line_bg.append(bg + (8 if (blink and ice) else 0))
        i += 1
    if line_ch:
        grid.append((line_ch, line_fg, line_bg))

    rows = rows or len(grid)
    ch = np.full((rows, cols), 0x20, np.uint8)
    fga = np.zeros((rows, cols), np.uint8)
    bga = np.zeros((rows, cols), np.uint8)
    for y, (lc, lf, lb) in enumerate(grid[:rows]):
        n = min(cols, len(lc))
        ch[y, :n] = lc[:n]
        fga[y, :n] = lf[:n]
        bga[y, :n] = lb[:n]
    return ch, fga, bga
