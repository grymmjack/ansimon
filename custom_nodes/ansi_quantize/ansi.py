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

# CGA palette order and ANSI SGR order are NOT the same — the RGB bits run the
# other way round. Palette index 1 is BLUE, but SGR 31 is RED; index 3 is CYAN,
# SGR 33 is YELLOW. Writing `30 + index` silently swaps red<->blue and
# cyan<->brown in every file produced.
#
#   palette:  0 black  1 blue   2 green  3 cyan  4 red   5 magenta 6 brown 7 grey
#   SGR:     30 black 31 red   32 green 33 yellow 34 blue 35 magenta 36 cyan 37 white
CGA_TO_SGR = (0, 4, 2, 6, 1, 5, 3, 7)
SGR_TO_CGA = (0, 4, 2, 6, 1, 5, 3, 7)      # the mapping is its own inverse
SGR_FG = {i: 30 + CGA_TO_SGR[i] for i in range(8)}
SGR_BG = {i: 40 + CGA_TO_SGR[i] for i in range(8)}

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
CELLS_MAGIC = b"ACEL\x03"
_DEPTH_CODE = {"16": 0, "256": 1, "rgb": 2}
_CODE_DEPTH = {v: k for k, v in _DEPTH_CODE.items()}


def pack_cells(ch, fg, bg, palette, ice, cell_h=16, cell_w=8, depth="16"):
    """Cell grid + palette -> compact bytes for the ComfyUI STRING channel.

    At depth `rgb` the fg/bg planes are (rows, cols, 3) instead of (rows, cols),
    so the blob is three times as wide there. Only the 16 base colours travel:
    the 256-colour table is derived from them, and truecolor needs no table.
    """
    ch = np.asarray(ch, np.uint8)
    rows, cols = ch.shape
    depth = str(depth)
    out = bytearray(CELLS_MAGIC)
    out += int(cols).to_bytes(2, "little")
    out += int(rows).to_bytes(2, "little")
    out.append(1 if ice else 0)
    out.append(int(cell_h) & 0xFF)
    out.append(int(cell_w) & 0xFF)
    out.append(_DEPTH_CODE[depth])
    for r, g, b in list(palette)[:16]:
        out += bytes((int(r) & 0xFF, int(g) & 0xFF, int(b) & 0xFF))
    out += ch.tobytes()
    out += np.asarray(fg, np.uint8).tobytes()
    out += np.asarray(bg, np.uint8).tobytes()
    return bytes(out)


def unpack_cells(data):
    """Inverse of pack_cells -> (ch, fg, bg, palette, ice, cell_h, cell_w, depth)."""
    if data[:5] != CELLS_MAGIC:
        raise ValueError("not an ansimon cell blob")
    cols = int.from_bytes(data[5:7], "little")
    rows = int.from_bytes(data[7:9], "little")
    ice = bool(data[9])
    cell_h, cell_w = int(data[10]), int(data[11])
    depth = _CODE_DEPTH[data[12]]
    base = 13
    pal = [tuple(data[base + i * 3:base + 3 + i * 3]) for i in range(16)]
    n = rows * cols
    off = base + 48
    w = 3 if depth == "rgb" else 1
    a = np.frombuffer(data, np.uint8, count=n * (1 + 2 * w), offset=off)
    shape = (rows, cols, 3) if depth == "rgb" else (rows, cols)
    return (a[:n].reshape(rows, cols).copy(),
            a[n:n + n * w].reshape(shape).copy(),
            a[n + n * w:].reshape(shape).copy(),
            pal, ice, cell_h, cell_w, depth)


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


# ---------------------------------------------------------------------------
# Colour depth: how a cell's two colours get written into the stream.
#
# The three encoders below all answer the same two questions — "what attribute
# does this cell want" and "what bytes select it" — so `to_ans` can stay a
# single function. That matters more than it looks: the deferred-wrap handling
# in `to_ans` took three bugs to get right, and forking it per colour depth
# would have been three chances to get it wrong again.
#
#   16   ESC[1;36;44m          the ANSI.SYS dialect every viewer speaks
#   256  ESC[38;5;n;48;5;nm    xterm indexed; the low 16 are the VIEWER's
#   rgb  ESC[1;r;g;bt          PabloDraw/SyncTERM 24-bit — sel 1 = fg, 0 = bg
#
# The 24-bit form uses `t`, not `m`. That is not a typo and not xterm's
# `38;2;r;g;b` — it is the scene's own extension, which is what PabloDraw
# writes and what IMG2ANS emits. We write an SGR approximation alongside it so
# a viewer that ignores `t` still shows recognisable art instead of one colour.
# ---------------------------------------------------------------------------
DEPTHS = ("16", "256", "rgb")


class _Attr16:
    """Classic 16-colour attributes. fg/bg are palette indices 0-15."""
    kind = "16"

    def __init__(self, fg, bg, ice=False):
        self.fg, self.bg, self.ice = fg, bg, ice

    def key(self, y, x):
        return int(self.fg[y][x]), int(self.bg[y][x])

    def blank_bg(self, y, x):
        return int(self.bg[y][x]) == 0

    def space_fg(self, key, cur):
        # A space shows only its background, so reuse the live foreground and
        # avoid an attribute change that would buy nothing.
        return (cur[0], key[1]) if cur else key

    def seq(self, key, cur):
        f, b = key
        out = b""
        # There is no "unbold" in this dialect, so dropping a bright colour
        # means a full reset before re-selecting.
        if cur is not None and (cur[0] >= 8 > f or (self.ice and cur[1] >= 8 > b)):
            out += f"{ESC}[0m".encode("ascii")
        return out + _sgr(f, b, self.ice).encode("ascii")


class _Attr256:
    """xterm indexed colour. fg/bg are indices 0-255, backgrounds unrestricted."""
    kind = "256"

    def __init__(self, fg, bg, **_):
        self.fg, self.bg = fg, bg

    def key(self, y, x):
        return int(self.fg[y][x]), int(self.bg[y][x])

    def blank_bg(self, y, x):
        return int(self.bg[y][x]) == 0

    def space_fg(self, key, cur):
        return (cur[0], key[1]) if cur else key

    def seq(self, key, cur):
        # Every sequence is absolute, so no reset dance is needed.
        return f"{ESC}[38;5;{key[0]};48;5;{key[1]}m".encode("ascii")


class _AttrRGB:
    """24-bit colour. fg/bg are (rows, cols, 3) uint8 arrays.

    `dialect` picks the wire format: `pablo` writes the scene's `ESC[<sel>;r;g;bt`
    pair (plus a 16-colour SGR fallback), `xterm` writes `ESC[38;2;r;g;bm`.

    `fallback` is the `(fg_map, bg_map)` pair from `nodes.rgb_fallback` — RGB
    tuple to nearest 16-colour index, one table per plane.
    """
    kind = "rgb"

    def __init__(self, fg, bg, dialect="pablo", fallback=None, ice=True, **_):
        self.fg = np.asarray(fg, np.uint8)
        self.bg = np.asarray(bg, np.uint8)
        self.dialect = dialect
        self.ice = ice
        # Nearest 16-colour index per cell, for the SGR fallback. Computed by
        # the caller against whatever palette the piece was built for.
        self.fallback = fallback

    def key(self, y, x):
        return (tuple(int(v) for v in self.fg[y][x]),
                tuple(int(v) for v in self.bg[y][x]))

    def blank_bg(self, y, x):
        return not self.bg[y][x].any()

    def space_fg(self, key, cur):
        return (cur[0], key[1]) if cur else key

    def seq(self, key, cur):
        (fr, fg_, fb), (br, bg_, bb) = key
        if self.dialect == "xterm":
            return (f"{ESC}[38;2;{fr};{fg_};{fb};"
                    f"48;2;{br};{bg_};{bb}m").encode("ascii")
        out = b""
        if self.fallback is not None:
            fmap, bmap = self.fallback
            f16, b16 = fmap.get(key[0]), bmap.get(key[1])
            if f16 is not None and b16 is not None:
                out += _sgr(f16, b16, self.ice).encode("ascii")
        return (out
                + f"{ESC}[1;{fr};{fg_};{fb}t".encode("ascii")
                + f"{ESC}[0;{br};{bg_};{bb}t".encode("ascii"))


def make_attr(fg, bg, depth="16", ice=False, dialect="pablo", fallback=None):
    """Pick the encoder for a colour depth."""
    depth = str(depth)
    if depth == "256":
        return _Attr256(fg, bg)
    if depth == "rgb":
        return _AttrRGB(fg, bg, dialect=dialect, fallback=fallback, ice=ice)
    return _Attr16(fg, bg, ice=ice)


def to_ans(ch, fg, bg, ice=False, width=None, trim_trailing=True,
           depth="16", dialect="pablo", fallback=None):
    """Serialise a cell grid to `.ANS` bytes (CP437, no SAUCE record).

    Attribute changes are emitted only when they actually change, which is
    what keeps real ANSI files small. The catch is that turning a bright
    foreground *off* needs a full reset (`ESC[0m`) — there is no "unbold" in
    the ANSI.SYS dialect these files target — so we reset and re-issue rather
    than trying to diff attribute-by-attribute.

    A row that fills the full width gets NO line ending. The terminal wraps by
    itself at the last column, so an explicit CRLF on top of that advances a
    second time and leaves a blank row. Get this wrong and a piece whose rows
    are all full-width comes out at double height — which is what happened
    here, and it is invisible unless you render the .ans with something other
    than your own code.
    """
    ch = np.asarray(ch, np.uint8)
    at = make_attr(np.asarray(fg), np.asarray(bg), depth=depth, ice=ice,
                   dialect=dialect, fallback=fallback)
    rows, cols = ch.shape
    width = width or cols

    out = bytearray()
    cur = None                                   # currently-active attribute

    # A full-width FINAL row is the one case auto-wrap cannot express. Writing
    # its last column leaves the terminal in a pending wrap, and a renderer
    # that resolves that eagerly gains a phantom blank row at the bottom.
    # Nothing emitted afterwards fixes it — not CR, not a reset — because the
    # wrap already happened. So write that last cell FIRST, by absolute
    # position, then draw the rest of the row; the final character printed is
    # then at column cols-1 and no wrap is pending.
    last_cell = None
    if rows and cols:
        lr = ch[rows - 1]
        end_last = cols
        if trim_trailing:
            while end_last > 0 and lr[end_last - 1] in (0x00, 0x20) \
                    and at.blank_bg(rows - 1, end_last - 1):
                end_last -= 1
        if end_last == cols:
            c = int(lr[cols - 1]) or 0x20
            k = at.key(rows - 1, cols - 1)
            out += f"{ESC}[{rows};{cols}H".encode("ascii")
            out += at.seq(k, None)
            out.append(c)
            out += f"{ESC}[H".encode("ascii")
            cur = k
            last_cell = (rows - 1, cols - 1)

    for y in range(rows):
        row_ch = ch[y]

        # Trailing run of blank-on-black costs nothing to omit — the newline
        # gets us to the next row anyway.
        end = cols
        if trim_trailing:
            while end > 0 and row_ch[end - 1] in (0x00, 0x20) \
                    and at.blank_bg(y, end - 1):
                end -= 1

        for x in range(end):
            if last_cell == (y, x):
                break                            # already placed, and printing
                                                 # it again would re-arm the wrap
            c = int(row_ch[x])
            if c < 0x20:
                # Control codes are valid CP437 glyphs but not valid stream
                # bytes: 0x1B would start an escape sequence, 0x1A would end
                # the art. The quantizer already refuses to pick them (see
                # cp437.STREAM_UNSAFE), so reaching here means the cells came
                # from elsewhere — an XBin via --from-ans can legitimately hold
                # them. Substituting a space loses that glyph, which is the only
                # option a .ANS leaves; the .xb keeps it intact.
                c = 0x20
            k = at.key(y, x)
            # A space shows only its background, so its foreground is free —
            # inheriting the current fg avoids a pointless attribute change.
            if c == 0x20 and cur is not None:
                k = at.space_fg(k, cur)
            if cur != k:
                out += at.seq(k, cur)
                cur = k
            out.append(c)

        # end < cols means the row stopped short, so it needs an explicit
        # newline; end == cols means the terminal already wrapped for us.
        if y != rows - 1 and end < cols:
            out += b"\r\n"

    out += f"{ESC}[0m".encode("ascii")
    return bytes(out)


# SAUCE names the font in TInfoS, and that name is what selects the CELL SIZE.
# Viewers read the SECOND word: "VGA50" and "EGA43" mean an 8x8 cell, while
# "IBM VGA" — and even "IBM VGA 850", where 850 is a codepage — mean 8x16.
# The XBin fontsize byte does not drive this for .ANS at all.
FONT_8x16 = "IBM VGA"
FONT_8x8 = "IBM VGA50"


def font_name_for(cell_h):
    return FONT_8x8 if cell_h == 8 else FONT_8x16


def sauce_record(data_len, cols, rows, title="", author="", group="",
                 date="", ice=False, aspect_square=True,
                 datatype=DATATYPE_CHARACTER, filetype=FILETYPE_ANSI,
                 font=None, cell_h=16, cell_w=8, comments=None):
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
    tflags |= ((0x02 if cell_w == 9 else 0x01) << 1)   # 01 = 8-dot, 10 = 9-dot
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
    rec += bytes([len(comnt_lines(comments))])   # comment lines
    rec += bytes([tflags])
    rec += fixed(font or font_name_for(cell_h), 22)   # TInfoS = font name
    assert len(rec) == 128, len(rec)
    return bytes(rec)


# ---------------------------------------------------------------------------
# SAUCE comments (the COMNT block) — where the settings that made the file go.
#
# Layout of a commented file:
#
#     <art bytes> 0x1A "COMNT" <n x 64-byte lines> <128-byte SAUCE record>
#
# Each line is EXACTLY 64 bytes, space padded, no terminator, and the record's
# comment-count byte must agree or readers mis-seek and lose the SAUCE entirely.
# Max 255 lines because the count is one byte.
#
# This is the only standard place in a .ANS to record how it was made. Worth
# doing: the art is reproducible from (seed, model, settings), and without them
# written down a piece you like is a piece you cannot re-derive.
# ---------------------------------------------------------------------------
COMNT_ID = b"COMNT"
COMNT_LINE = 64
COMNT_MAX = 255


def comnt_lines(comments):
    """Normalise comments to a list of <=64-char strings, at most 255 of them.

    Long lines are wrapped rather than truncated, so a full prompt survives.
    """
    if not comments:
        return []
    if isinstance(comments, str):
        comments = comments.splitlines()
    out = []
    for c in comments:
        c = str(c).rstrip()
        if not c:
            out.append("")
            continue
        while c and len(out) < COMNT_MAX:
            out.append(c[:COMNT_LINE])
            c = c[COMNT_LINE:]
    return out[:COMNT_MAX]


def comnt_block(comments):
    """The COMNT block bytes, or b'' when there are no comments."""
    lines = comnt_lines(comments)
    if not lines:
        return b""
    out = bytearray(COMNT_ID)
    for line in lines:
        out += line.encode("cp437", "replace")[:COMNT_LINE].ljust(COMNT_LINE, b" ")
    return bytes(out)


def write_ans(path, ch, fg, bg, ice=False, sauce=True, title="", author="",
              group="", date="", aspect_square=True, cell_h=16,
              depth="16", dialect="pablo", fallback=None, comments=None):
    """Write a complete `.ANS` file (art + EOF marker + [COMNT] + SAUCE)."""
    body = to_ans(ch, fg, bg, ice=ice, depth=depth, dialect=dialect,
                  fallback=fallback)
    with open(path, "wb") as f:
        f.write(body)
        if sauce:
            rows, cols = np.asarray(ch).shape
            f.write(b"\x1a")
            # COMNT goes BETWEEN the EOF marker and the record, and the record's
            # count byte must match — sauce_record derives it from the same list.
            f.write(comnt_block(comments))
            f.write(sauce_record(len(body), cols, rows, title, author, group,
                                 date, ice, aspect_square, cell_h=cell_h,
                                 comments=comments))
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
                        fg = SGR_TO_CGA[v - 30]
                    elif 40 <= v <= 47:
                        bg = SGR_TO_CGA[v - 40]
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
