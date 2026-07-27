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
import re

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
# Reader — a real cursor model, not a line-by-line reader.
#
# `.ANS` is a recording of a terminal session, so reading one means emulating
# the terminal. That is not academic: in a survey of 112 hand-drawn scene
# files, `ESC[#C` (cursor forward) appeared in 111 of them, 8336 times total.
# It is the standard space-saver — instead of writing forty spaces you write
# `ESC[40C` and skip, leaving whatever was already on the canvas. A parser that
# ignores cursor movement doesn't lose a little fidelity; it collapses the art.
#
# `ESC[#A` (cursor up) showed up too, which means overdraw — draw a pass, go
# back up, draw over it. So the canvas has to be random-access, and cells have
# to be overwritable, which rules out the append-a-row-at-a-time approach.
# ---------------------------------------------------------------------------
DEFAULT_FG, DEFAULT_BG = 7, 0

_CSI = re.compile(rb"\x1b\[([0-9;]*)([@-~])")


class _Canvas:
    """A grid that grows downward on demand. ANSI art is a fixed width but an
    open-ended height — nothing in the file says how tall it is until it ends."""

    def __init__(self, cols):
        self.cols = cols
        self.ch, self.fg, self.bg = [], [], []

    def _ensure(self, y):
        while len(self.ch) <= y:
            self.ch.append([0x20] * self.cols)
            self.fg.append([DEFAULT_FG] * self.cols)
            self.bg.append([DEFAULT_BG] * self.cols)

    def put(self, y, x, c, f, b):
        if x < 0 or x >= self.cols or y < 0:
            return
        self._ensure(y)
        self.ch[y][x], self.fg[y][x], self.bg[y][x] = c, f, b

    def erase(self, mode, y, x):
        """ESC[#J — 0: to end of screen, 1: to start, 2: whole screen."""
        self._ensure(y)
        if mode == 2:
            for row in range(len(self.ch)):
                self.ch[row] = [0x20] * self.cols
                self.fg[row] = [DEFAULT_FG] * self.cols
                self.bg[row] = [DEFAULT_BG] * self.cols
        elif mode == 0:
            for xx in range(x, self.cols):
                self.ch[y][xx], self.fg[y][xx], self.bg[y][xx] = 0x20, DEFAULT_FG, DEFAULT_BG
            for row in range(y + 1, len(self.ch)):
                self.ch[row] = [0x20] * self.cols
                self.fg[row] = [DEFAULT_FG] * self.cols
                self.bg[row] = [DEFAULT_BG] * self.cols
        elif mode == 1:
            for row in range(0, y):
                self.ch[row] = [0x20] * self.cols
                self.fg[row] = [DEFAULT_FG] * self.cols
                self.bg[row] = [DEFAULT_BG] * self.cols
            for xx in range(0, min(x + 1, self.cols)):
                self.ch[y][xx], self.fg[y][xx], self.bg[y][xx] = 0x20, DEFAULT_FG, DEFAULT_BG

    def arrays(self, rows=None):
        n = rows or max(1, len(self.ch))
        self._ensure(n - 1)
        a = lambda src, d: np.array([r for r in src[:n]], np.uint8)
        return (a(self.ch, 0x20), a(self.fg, DEFAULT_FG), a(self.bg, DEFAULT_BG))


def read_sauce(data):
    """Return (sauce_dict_or_None, body_without_sauce)."""
    cut = data.rfind(b"\x1a")
    if cut == -1 or len(data) - cut - 1 < 128 or data[cut + 1:cut + 6] != SAUCE_ID:
        return None, data
    rec = data[cut + 1:cut + 129]
    return {
        "title": rec[7:42].decode("cp437", "replace").strip(),
        "author": rec[42:62].decode("cp437", "replace").strip(),
        "group": rec[62:82].decode("cp437", "replace").strip(),
        "date": rec[82:90].decode("cp437", "replace").strip(),
        "datatype": rec[94], "filetype": rec[95],
        "cols": int.from_bytes(rec[96:98], "little"),
        "rows": int.from_bytes(rec[98:100], "little"),
        "tflags": rec[105],
        "ice": bool(rec[105] & 0x01),
    }, data[:cut]


def parse_ans(data, cols=80, rows=None, ice=None, wrap=True):
    """Parse `.ANS` bytes -> (ch, fg, bg) uint8 arrays.

    Emulates the subset of ANSI.SYS that art actually uses: SGR, cursor
    movement (CUU/CUD/CUF/CUB), absolute positioning (CUP/HVP), and erase
    (ED/EL). Anything else is consumed and ignored rather than printed, which
    is what a terminal would do.

    `ice=None` means "believe the SAUCE record if there is one", so a file that
    declares non-blink gets 16 background colours and one that doesn't gets 8
    plus a blink bit — rather than us imposing a guess on someone's art.
    """
    sauce, body = read_sauce(data)
    if sauce:
        cols = sauce["cols"] or cols
        rows = rows or (sauce["rows"] or None)
        if ice is None:
            ice = sauce["ice"]
    if ice is None:
        ice = False

    cv = _Canvas(cols)
    x = y = 0
    fg, bg, bold, blink = DEFAULT_FG, DEFAULT_BG, False, False
    saved = (0, 0)
    i, n = 0, len(body)
    # Deferred wrap, exactly as a real terminal does it: after printing into
    # the last column the cursor STAYS there and only wraps when the next
    # printable character shows up. Wrapping eagerly double-advances on a full
    # row followed by CRLF, which silently drops every other line.
    pending_wrap = False

    while i < n:
        b = body[i]

        if b == 0x1B and i + 1 < n and body[i + 1] == ord("["):
            m = _CSI.match(body, i)
            if not m:
                i += 1
                continue
            params, final = m.group(1), m.group(2).decode()
            nums = [int(t) if t else 0 for t in params.split(b";")] if params else []
            p0 = nums[0] if nums else 0

            if final == "m":
                for v in (nums or [0]):
                    if v == 0:
                        fg, bg, bold, blink = DEFAULT_FG, DEFAULT_BG, False, False
                    elif v == 1:
                        bold = True
                    elif v in (5, 6):
                        blink = True
                    elif v == 7:
                        fg, bg = bg, fg
                    elif v in (21, 22):
                        bold = False
                    elif v == 25:
                        blink = False
                    elif 30 <= v <= 37:
                        fg = v - 30
                    elif 40 <= v <= 47:
                        bg = v - 40
            elif final == "A":
                y = max(0, y - max(1, p0))
            elif final == "B":
                y += max(1, p0)
            elif final == "C":
                x = min(cols - 1, x + max(1, p0))
            elif final == "D":
                x = max(0, x - max(1, p0))
            elif final in ("H", "f"):
                y = max(0, (nums[0] if len(nums) > 0 and nums[0] else 1) - 1)
                x = max(0, (nums[1] if len(nums) > 1 and nums[1] else 1) - 1)
            elif final == "J":
                cv.erase(p0, y, x)
            elif final == "K":
                cv._ensure(y)
                rng = (range(x, cols) if p0 == 0 else
                       range(0, min(x + 1, cols)) if p0 == 1 else range(cols))
                for xx in rng:
                    cv.ch[y][xx], cv.fg[y][xx], cv.bg[y][xx] = 0x20, DEFAULT_FG, DEFAULT_BG
            elif final == "s":
                saved = (x, y)
            elif final == "u":
                x, y = saved
            if final not in ("m",):
                pending_wrap = False        # any cursor motion cancels the wrap
            i = m.end()
            continue

        if b == 0x0D:
            x = 0
            pending_wrap = False
        elif b == 0x0A:
            y += 1
            x = 0
            pending_wrap = False
        elif b == 0x08:
            x = max(0, x - 1)
            pending_wrap = False
        elif b == 0x1A:
            break                                  # DOS EOF
        else:
            if pending_wrap:
                x, y = 0, y + 1
                pending_wrap = False
            cv.put(y, x, b, fg + (8 if bold else 0),
                   bg + (8 if (blink and ice) else 0))
            if x + 1 >= cols:
                if wrap:
                    pending_wrap = True            # wrap later, not now
                else:
                    x = cols - 1
            else:
                x += 1
        i += 1

    return cv.arrays(rows)
