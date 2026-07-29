"""AnsiQuantize — turn an SDXL render into real ANSI art, then draw it back.

Pipeline:

    render -> grid-align -> sample to cell resolution -> per-cell (glyph,fg,bg)
           -> render glyph bitmaps -> PNG        (+ the .ANS that produced it)

The important thing to understand is that the PNG is not a stylised filter over
the render. It is a *picture of the .ANS file* — every pixel comes from a CP437
glyph bitmap filled with one of 16 palette colours. If you open the emitted
.ANS in PabloDraw or Moebius you get the same image back. That is the whole
point of routing through a real cell grid instead of just posterising.

Two ideas are carried over from pixelmon's `pixelart_palette`, because they
solve the same problems one step earlier in the pipeline:

  * `flatten_shrink()` — the grid-aware downscale. Pixel Art XL (and the ANSI
    LoRA) paint chunky blocks at 1024px, i.e. a much smaller *logical* image.
    Recovering that logical grid with a ModeFilter before reducing avoids
    sampling mid-block noise. Sampling straight down to 80x25 without it gives
    you mush.
  * `nearest_indices()` — redmean perceptual colour distance rather than plain
    RGB Euclidean. This matters far more at 16 colours than at 32: a wrong pick
    is 1/16th of the entire available gamut.

Outputs:
    ansi     — the rendered art, cols*8 x rows*16 px. SaveImage this.
    preview  — the same image scaled up / aspect-corrected for eyeballing.
    ans_b64  — the real .ANS file, base64'd so it survives a trip across the
               render farm as a ComfyUI STRING.
"""
import base64
import os
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image, ImageFilter

from . import ansi as ansi_fmt
from . import xbin as xbin_fmt
from .cp437 import CELL_H, CELL_W, CHARSETS, charset_indices, glyph_bitmaps
from .palette import ALL_PALETTES, parse_palette

_RESAMPLE = {"nearest": Image.NEAREST, "box (area average)": Image.BOX,
             "lanczos": Image.LANCZOS}
_CHARSET_NAMES = list(CHARSETS.keys())


# ---------------------------------------------------------------------------
# ComfyUI IMAGE tensors are float32 [B,H,W,C] in [0,1].
# ---------------------------------------------------------------------------
def _tensor_to_pil(img):
    arr = (img[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _pil_to_tensor(pil):
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ]


# ---------------------------------------------------------------------------
# Perceptual colour distance (redmean). Ported from pixelmon's pixelart_palette
# so both projects pick the same swatch for the same colour.
# ---------------------------------------------------------------------------
def nearest_indices(pixels, palette):
    px = np.asarray(pixels, np.float64)
    pal = np.asarray(palette, np.float64)
    rmean = (px[:, None, 0] + pal[None, :, 0]) * 0.5
    dr = px[:, None, 0] - pal[None, :, 0]
    dg = px[:, None, 1] - pal[None, :, 1]
    db = px[:, None, 2] - pal[None, :, 2]
    dist = ((2 + rmean / 256.0) * dr * dr
            + 4 * dg * dg
            + (2 + (255 - rmean) / 256.0) * db * db)
    return dist.argmin(axis=1)


# ---------------------------------------------------------------------------
# Grid alignment — pixelmon's trick, applied before we hit the character grid.
# ---------------------------------------------------------------------------
def flatten_shrink(src, target_long, resample=Image.BOX, smooth="mode"):
    """Recover the render's native block grid, then reduce to `target_long`.

    The ModeFilter collapses each painted block to its dominant colour, so the
    subsequent resize samples clean block centres instead of the blurry seams
    between them.
    """
    sw, sh = src.size
    if smooth != "none" and max(sw, sh) > target_long:
        block = max(3, int(round(max(sw, sh) / target_long)) | 1)      # odd >= 3
        fil = ImageFilter.ModeFilter if smooth == "mode" else ImageFilter.MedianFilter
        src = src.filter(fil(size=block))
    scl = target_long / max(sw, sh)
    return src.resize((max(1, round(sw * scl)), max(1, round(sh * scl))),
                      resample=resample)


def _snapper_bin():
    """Locate pixelmon's spritefusion-pixel-snapper, if this box has one."""
    env = os.environ.get("ANSIMON_SNAPPER") or os.environ.get("PIXELMON_SNAPPER")
    if env and os.path.exists(env):
        return env
    for root in (os.path.expanduser("~/pixelmon"),
                 os.path.expanduser("~/git/pixelmon")):
        cand = os.path.join(root, "tools", "pixel-snapper", "target", "release",
                            "spritefusion-pixel-snapper")
        if os.path.exists(cand):
            return cand
    return None


def _snap_pixels(pil_img, k_colors):
    """Run pixelmon's Rust pixel-snapper to auto-detect and lock the block grid."""
    binp = _snapper_bin()
    if not binp:
        raise RuntimeError(
            "pixel-snapper not found. It ships with pixelmon — build it once:\n"
            "  cd ~/pixelmon/tools/pixel-snapper && cargo build --release\n"
            "or point $ANSIMON_SNAPPER at the binary. (Optional: leave "
            "snap_pixels off to use the ModeFilter grid recovery instead.)")
    with tempfile.TemporaryDirectory() as td:
        ip, op = os.path.join(td, "in.png"), os.path.join(td, "out.png")
        pil_img.convert("RGB").save(ip)
        subprocess.run([binp, ip, op, str(int(k_colors))],
                       check=True, capture_output=True, timeout=180)
        return Image.open(op).convert("RGB").copy()


# ---------------------------------------------------------------------------
# The cell matcher — the heart of the whole thing.
#
# For one cell we must choose a character `g`, a foreground colour and a
# background colour so that drawing `g` in those colours best approximates the
# source pixels. Given a candidate glyph, the OPTIMAL colours are simply the
# mean of the pixels the glyph covers (fg) and the mean of the pixels it does
# not (bg) — so we only ever have to search over characters, not over the
# 16x16 colour pairs, which is what makes this tractable.
#
# The squared error for a candidate expands to a form built entirely out of
# per-(cell, glyph) sums, so every glyph for every cell is evaluated in two
# matrix multiplies rather than a Python loop.
# ---------------------------------------------------------------------------
class _CellStats:
    """Precomputed per-(cell, glyph) sums used to score candidates."""

    def __init__(self, patches, masks):
        n, p, _ = patches.shape
        g = masks.shape[0]
        mf = masks.astype(np.float64)                       # (G,P)
        self.n, self.p, self.g = n, p, g
        self.cnt = mf.sum(1)                                # (G,)
        self.cnt_bg = p - self.cnt

        # Sum of pixel values under / outside each mask, via one BLAS matmul.
        a = patches.transpose(0, 2, 1).reshape(n * 3, p)    # (N*3, P)
        self.sm = (a @ mf.T).reshape(n, 3, g).transpose(0, 2, 1)     # (N,G,3)
        self.s_total = patches.sum(1)                       # (N,3)
        self.sb = self.s_total[:, None, :] - self.sm        # (N,G,3)

        a2 = (patches ** 2).transpose(0, 2, 1).reshape(n * 3, p)
        self.ssm = (a2 @ mf.T).reshape(n, 3, g).sum(1)      # (N,G)
        self.ss_total = (patches ** 2).sum(axis=(1, 2))     # (N,)

        # Guard the degenerate glyphs: full block has no background pixels,
        # space has no foreground pixels.
        self.inv_cnt = np.where(self.cnt > 0, 1.0 / np.maximum(self.cnt, 1), 0.0)
        self.inv_cnt_bg = np.where(self.cnt_bg > 0,
                                   1.0 / np.maximum(self.cnt_bg, 1), 0.0)


def _score(st, pal, n_bg, rows_slice=None, bias=None):
    """Score every glyph for the given cells -> (err, fg_idx, bg_idx)."""
    sm = st.sm if rows_slice is None else st.sm[rows_slice]
    sb = st.sb if rows_slice is None else st.sb[rows_slice]
    ssm = st.ssm if rows_slice is None else st.ssm[rows_slice]
    ss_total = st.ss_total if rows_slice is None else st.ss_total[rows_slice]

    if bias is not None:
        # Shift this cell's colour statistics by the diffused error. A constant
        # per-pixel shift of `d` moves the masked sum by cnt*d.
        sm = sm + st.cnt[None, :, None] * bias[:, None, :]
        sb = sb + st.cnt_bg[None, :, None] * bias[:, None, :]

    n, g = sm.shape[0], st.g
    fg_mean = sm * st.inv_cnt[None, :, None]
    bg_mean = sb * st.inv_cnt_bg[None, :, None]

    fg_idx = nearest_indices(fg_mean.reshape(-1, 3), pal).reshape(n, g)
    bg_idx = nearest_indices(bg_mean.reshape(-1, 3), pal[:n_bg]).reshape(n, g)

    cf = pal[fg_idx]                                        # (n,G,3)
    cb = pal[bg_idx]
    err = (ssm - 2.0 * (cf * sm).sum(-1) + st.cnt[None, :] * (cf ** 2).sum(-1)
           + (ss_total[:, None] - ssm)
           - 2.0 * (cb * sb).sum(-1) + st.cnt_bg[None, :] * (cb ** 2).sum(-1))
    return err, fg_idx, bg_idx


def shade_blends(pal, n_bg, shades=(0xB0, 0xB1, 0xB2), allow=None):
    """Every colour a shade character can produce -> (blend_rgb, glyph, fg, bg).

    This is how ANSI actually dithers. `0xB0 0xB1 0xB2` cover 25/50/75% of the
    cell, so drawing one in foreground colour F over background B yields a
    perceived colour `cov*F + (1-cov)*B` — an intermediate tone that is not in
    the 16-colour palette at all. Three shades x 16 x 16 pairs turn 16 colours
    into a few hundred usable tones.

    The matcher cannot find these by deriving fg/bg from the cell's own pixels:
    on a flat cell both derived means are the same value and quantize to the
    same entry, so the shade renders solid. The blends have to be enumerated.
    """
    cov = {0xB0: 0.25, 0xB1: 0.50, 0xB2: 0.75}
    ok = set(range(16)) if not allow else set(allow)
    out_rgb, out_meta = [], []
    for g in shades:
        c = cov[g]
        for f in range(16):
            if f not in ok:
                continue
            for b in range(n_bg):
                if f == b or b not in ok:
                    continue                      # renders solid; already covered
                out_rgb.append(c * pal[f] + (1.0 - c) * pal[b])
                out_meta.append((g, f, b))
    return np.asarray(out_rgb), out_meta


def parse_colors(spec):
    """'3,8,15,11' -> sorted tuple of palette indices, or None for all 16."""
    if not spec or not str(spec).strip():
        return None
    out = set()
    for tok in str(spec).replace(",", " ").split():
        try:
            v = int(tok, 0)
        except ValueError:
            raise ValueError(f"--colors: {tok!r} is not a palette index 0-15")
        if not 0 <= v <= 15:
            raise ValueError(f"--colors: {v} out of range (0-15)")
        out.add(v)
    if 0 not in out:
        out.add(0)          # black is the canvas; without it nothing can be empty
    return tuple(sorted(out))


def match_cells(patches, chars, pal, ice=False, dither=False, cols=None,
                strength=0.75, clamp=64.0, portable_bias=0.5, shade_blend=True,
                shade_bias=0.10, allow=None):
    """Choose (char, fg, bg) for every cell.

    `patches` is (N, CELL_H*CELL_W, 3) float in 0-255, `chars` the allowed
    CP437 indices. With `dither`, residual error is diffused to neighbouring
    cells in serpentine order — the cell-level equivalent of Floyd-Steinberg,
    which buys back a lot of gradient at 16 colours.

    `strength` and `clamp` tame that diffusion. Undamped error diffusion into
    a 16-colour palette drifts: the residual accumulates across all three
    channels until it tips a near-grey cell into a saturated hue, which shows
    up as coloured confetti in flat areas. Damping to 0.75 and clamping the
    carried error to +/-64 keeps the useful gradient and drops the confetti.

    `portable_bias` breaks ties toward the more portable encoding. A solid
    bright cell can be written as a space with a bright BACKGROUND (needs iCE
    colours) or as a full block with a bright FOREGROUND (works in every
    viewer ever made). Both render identically here, so we nudge the matcher
    toward the one that survives contact with other people's software.
    """
    masks = glyph_bitmaps()[list(chars)].reshape(len(chars), -1)      # (G,P)
    st = _CellStats(patches, masks)
    pal = np.asarray(pal, np.float64)
    n_bg = 16 if ice else 8
    chars = np.asarray(chars)

    # Restricting the palette is done by pushing the disallowed entries far
    # away in colour space rather than by reindexing, so every downstream
    # index still means what it means in a .ANS file. A picked colour is
    # always a real ANSI attribute, just never one you excluded.
    if allow:
        keep = np.zeros(16, bool)
        keep[list(allow)] = True
        pal = pal.copy()
        pal[~keep] = 1e6

    # Scale the tie-break to the error units in play (sum of squared channel
    # error over 128 pixels), so it only ever decides genuine ties.
    tie = portable_bias * st.p

    if not dither:
        err, fg_idx, bg_idx = _score(st, pal, n_bg)
        err = err + tie * (bg_idx >= 8)
        best = err.argmin(1)
        rows = np.arange(st.n)
        out_ch = chars[best]
        out_fg = fg_idx[rows, best]
        out_bg = bg_idx[rows, best]
        out_cnt = st.cnt[best]
        out_err = err[rows, best]

        if shade_blend and any(c in (0xB0, 0xB1, 0xB2) for c in chars):
            # Offer every shade blend as an alternative for each cell and take
            # it when it reproduces the cell's average colour better. Scored on
            # the CELL MEAN, because at 8x16 px the eye integrates the cell —
            # which is exactly why a dither reads as an intermediate tone
            # rather than as a pattern.
            blends, meta = shade_blends(pal, n_bg, allow=allow)
            mean = st.s_total / st.p                              # (N,3)
            bi = nearest_indices(mean, blends)                    # (N,)
            cand = blends[bi]
            shade_err = st.p * ((cand - mean) ** 2).sum(-1)
            # what the current pick averages out to
            cur = (pal[out_fg] * out_cnt[:, None]
                   + pal[out_bg] * (st.p - out_cnt)[:, None]) / st.p
            cur_err = st.p * ((cur - mean) ** 2).sum(-1)
            # `shade_bias` is how much BETTER a blend must be to displace the
            # structural pick. At 1.0 a blend wins on any improvement, which
            # dithers the whole canvas — measured at 75% shades against a real
            # corpus's 10%. Requiring a clear win keeps shades where they
            # belong: gradients, over a structure of solids and half blocks.
            take = shade_err < cur_err * shade_bias
            for i in np.where(take)[0]:
                g, f, b = meta[bi[i]]
                out_ch[i], out_fg[i], out_bg[i] = g, f, b
                out_cnt[i] = {0xB0: 0.25, 0xB1: 0.50, 0xB2: 0.75}[g] * st.p
        return out_ch, out_fg, out_bg, out_cnt, out_err

    # --- serpentine cell-level error diffusion --------------------------------
    assert cols, "dither needs the grid width to walk rows"
    n_rows = st.n // cols
    bias = np.zeros((st.n, 3))
    out_ch = np.zeros(st.n, np.int64)
    out_fg = np.zeros(st.n, np.int64)
    out_bg = np.zeros(st.n, np.int64)
    out_cnt = np.zeros(st.n)

    mean_actual = st.s_total / st.p                          # (N,3)
    for y in range(n_rows):
        xs = range(cols) if y % 2 == 0 else range(cols - 1, -1, -1)
        for x in xs:
            i = y * cols + x
            e, fi, bi = _score(st, pal, n_bg, rows_slice=slice(i, i + 1),
                               bias=bias[i:i + 1])
            e = e + tie * (bi >= 8)
            b = int(e[0].argmin())
            out_ch[i], out_fg[i], out_bg[i] = chars[b], fi[0, b], bi[0, b]
            out_cnt[i] = st.cnt[b]

            # Residual = what the cell should have averaged vs what we drew.
            drawn = (pal[fi[0, b]] * st.cnt[b] + pal[bi[0, b]] * st.cnt_bg[b]) / st.p
            resid = ((mean_actual[i] + bias[i]) - drawn) * strength
            fwd = 1 if y % 2 == 0 else -1
            for dx, dy, w in ((fwd, 0, 7 / 16), (-fwd, 1, 3 / 16),
                              (0, 1, 5 / 16), (fwd, 1, 1 / 16)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < n_rows:
                    j = ny * cols + nx
                    bias[j] = np.clip(bias[j] + resid * w, -clamp, clamp)
    return out_ch, out_fg, out_bg, out_cnt, None


# ---------------------------------------------------------------------------
# Rendering: paint the chosen cells back out through the same glyph bitmaps.
# ---------------------------------------------------------------------------
def render_cells(ch, fg, bg, pal):
    """(rows,cols) cell arrays -> (rows*16, cols*8, 3) uint8 image."""
    rows, cols = ch.shape
    glyphs = glyph_bitmaps()                                 # (256,16,8) bool
    pal = np.asarray(pal, np.uint8)
    masks = glyphs[ch]                                       # (rows,cols,16,8)
    fg_col = pal[fg][:, :, None, None, :]                    # (rows,cols,1,1,3)
    bg_col = pal[bg][:, :, None, None, :]
    cell = np.where(masks[..., None], fg_col, bg_col)        # (rows,cols,16,8,3)
    return (cell.transpose(0, 2, 1, 3, 4)
                .reshape(rows * CELL_H, cols * CELL_W, 3)
                .astype(np.uint8))


def apply_aspect(pil, mode):
    """Simulate a CRT's non-square text-mode pixels.

    80x25 VGA text was 720x400 shown on a 4:3 screen, so the pixels were taller
    than they were wide. Our cells are 8px, giving a 640x400 canvas; stretching
    height by 1.2 to 640x480 is what makes it look the way it did on the CRT
    instead of squashed.
    """
    if mode == "square":
        return pil
    w, h = pil.size
    return pil.resize((w, int(round(h * 1.2))), Image.NEAREST)


# ---------------------------------------------------------------------------
class AnsiQuantize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cols": ("INT", {"default": 80, "min": 8, "max": 320, "step": 1}),
                "rows": ("INT", {"default": 50, "min": 4, "max": 200, "step": 1}),
                "charset": (list(_CHARSET_NAMES), {"default": "halfblock"}),
                "palette": (list(ALL_PALETTES.keys()) + ["Custom"],
                            {"default": "ansi"}),
            },
            "optional": {
                "ice_colors": ("BOOLEAN", {"default": True}),
                "dither": ("BOOLEAN", {"default": False}),
                "dither_strength": ("FLOAT", {"default": 0.75, "min": 0.0,
                                              "max": 1.0, "step": 0.05}),
                "colors": ("STRING", {"default": ""}),
                "shading": (["none", "light", "medium", "full"],
                            {"default": "light"}),
                "smooth": (["mode", "median", "none"], {"default": "mode"}),
                "pixel_grid": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 8}),
                "snap_pixels": ("BOOLEAN", {"default": False}),
                "snap_colors": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "aspect": (["square", "classic (4:3)"], {"default": "square"}),
                "view_scale": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "scale": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "resample": (list(_RESAMPLE.keys()), {"default": "box (area average)"}),
                "force_black_bg": ("BOOLEAN", {"default": False}),
                "custom_hex": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("ansi", "preview", "cells_b64")
    FUNCTION = "process"
    CATEGORY = "image/ansi art"

    def process(self, image, cols, rows, charset, palette,
                ice_colors=True, dither=False, dither_strength=0.75,
                colors="", shading="light",
                smooth="mode", pixel_grid=0,
                snap_pixels=False, snap_colors=0, aspect="square", view_scale=1, scale=1,
                resample="box (area average)", force_black_bg=False,
                custom_hex=""):
        pal = np.asarray(parse_palette(palette, custom_hex), np.float64)
        chars = charset_indices(charset)
        pil = _tensor_to_pil(image)

        # --- 1. grid alignment ------------------------------------------------
        # Recover the render's own block grid before we impose the character
        # grid on it, so cell sampling lands on block centres.
        if snap_pixels:
            pil = _snap_pixels(pil, snap_colors or 64)
        elif pixel_grid:
            pil = flatten_shrink(pil, pixel_grid, Image.NEAREST, smooth)
        else:
            # Default: flatten at roughly twice the cell grid, which keeps
            # sub-cell detail alive for the glyph matcher while still killing
            # block-seam noise.
            pil = flatten_shrink(pil, max(cols, rows) * 4, Image.BOX, smooth)

        # --- 2. sample to exact cell resolution -------------------------------
        target = (cols * CELL_W, rows * CELL_H)
        arr = np.asarray(pil.resize(target, _RESAMPLE[resample]).convert("RGB"),
                         np.float64)

        # --- 3. per-cell character + colour choice ----------------------------
        patches = (arr.reshape(rows, CELL_H, cols, CELL_W, 3)
                      .transpose(0, 2, 1, 3, 4)
                      .reshape(rows * cols, CELL_H * CELL_W, 3))
        # How eagerly a shade blend may displace a solid pick. Tuned against a
        # real corpus, which sits near 10% shade characters: bias 1.0 gave 75%
        # (the whole canvas dithered), 0.35 gave 54%.
        bias = {"none": 0.0, "light": 0.10, "medium": 0.35, "full": 1.0}[shading]
        ch, fg, bg, cnt, _ = match_cells(patches, chars, pal, ice=ice_colors,
                                         dither=dither, cols=cols,
                                         strength=dither_strength,
                                         shade_blend=bias > 0, shade_bias=bias,
                                         allow=parse_colors(colors))

        ch = ch.reshape(rows, cols).astype(np.uint8)
        fg = fg.reshape(rows, cols).astype(np.uint8)
        bg = bg.reshape(rows, cols).astype(np.uint8)
        cnt = cnt.reshape(rows, cols)

        # Tidy the degenerate cases so the .ANS is clean: a full block has no
        # visible background, a blank has no visible foreground.
        bg[cnt >= CELL_H * CELL_W] = 0
        fg[cnt <= 0] = 7
        if force_black_bg:
            # Re-solve with background pinned to black — the look of a lot of
            # BBS-era work, where the canvas was simply the terminal.
            bg[:] = 0

        # --- 4. render the cells back out -------------------------------------
        out = render_cells(ch, fg, bg, pal.astype(np.uint8))
        img = Image.fromarray(out, "RGB")
        if scale > 1:
            # Integer nearest-neighbour only. The PNG stays a faithful picture
            # of the .ANS — every pixel still comes from a glyph bitmap, just
            # drawn larger. Any non-integer or smooth resize would break that.
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

        preview = apply_aspect(img, aspect)
        if view_scale > 1:
            preview = preview.resize((preview.width * view_scale,
                                      preview.height * view_scale), Image.NEAREST)

        cells = ansi_fmt.pack_cells(ch, fg, bg, pal.astype(np.uint8), ice_colors)
        return (_pil_to_tensor(img), _pil_to_tensor(preview),
                base64.b64encode(cells).decode("ascii"))


class SaveAnsi:
    """Serialise the cell grid to .ANS and/or .XB, on disk and inline.

    Two delivery routes on purpose. The file on disk is what you want when
    rendering locally; the base64 echo is what makes the render farm work,
    because ComfyUI only ever surfaces OUTPUT nodes in `/history` and a remote
    box's filesystem isn't ours to read. Shipping the bytes inline means a farm
    node needs no shared mount and no second HTTP round-trip.

    Format notes:
      * `.ans` is the portable one, but it is a stream with no width field, so
        anything other than 80 columns depends on the SAUCE record being read.
      * `.xb` (XBin) puts width/height in the header and can carry the font and
        palette with it. For non-80-column canvases — game placeholder art at
        whatever cell size you need — it is the format that actually works.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cells_b64": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "ansimon/art"}),
                "format": (["ans", "xb", "both"], {"default": "ans"}),
            },
            "optional": {
                "title": ("STRING", {"default": ""}),
                "author": ("STRING", {"default": ""}),
                "group": ("STRING", {"default": ""}),
                "date": ("STRING", {"default": ""}),
                "sauce": ("BOOLEAN", {"default": True}),
                "xb_compress": ("BOOLEAN", {"default": True}),
                "xb_embed_font": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "image/ansi art"
    OUTPUT_NODE = True

    def save(self, cells_b64, filename_prefix, format="ans", title="", author="",
             group="", date="", sauce=True, xb_compress=True, xb_embed_font=True):
        ch, fg, bg, pal, ice = ansi_fmt.unpack_cells(base64.b64decode(cells_b64))
        rows, cols = ch.shape

        blobs = {}
        if format in ("ans", "both"):
            body = ansi_fmt.to_ans(ch, fg, bg, ice=ice)
            if sauce:
                body += b"\x1a" + ansi_fmt.sauce_record(
                    len(body), cols, rows, title, author, group, date, ice)
            blobs["ans"] = body
        if format in ("xb", "both"):
            body = xbin_fmt.to_xbin(ch, fg, bg, palette=pal, ice=ice,
                                    compress=xb_compress, embed_font=xb_embed_font)
            if sauce:
                # SAUCE for XBin: DataType 6 (XBin), FileType 0. Width/height
                # are already authoritative in the XBin header; SAUCE just adds
                # the title/author/group metadata.
                body += b"\x1a" + ansi_fmt.sauce_record(
                    len(body), cols, rows, title, author, group, date, ice,
                    datatype=6, filetype=0)
            blobs["xb"] = body

        try:
            import folder_paths
            outdir = folder_paths.get_output_directory()
        except Exception:
            outdir = os.path.join(os.path.expanduser("~/ComfyUI"), "output")

        sub = os.path.dirname(filename_prefix)
        base = os.path.basename(filename_prefix) or "art"
        dest = os.path.join(outdir, sub)
        os.makedirs(dest, exist_ok=True)

        # One counter across both formats, so a .ans and its .xb twin share a
        # number instead of drifting apart run to run.
        n = 1
        while any(os.path.exists(os.path.join(dest, f"{base}_{n:05d}.{e}"))
                  for e in blobs):
            n += 1

        meta, payload = [], []
        for ext, body in blobs.items():
            fname = f"{base}_{n:05d}.{ext}"
            with open(os.path.join(dest, fname), "wb") as f:
                f.write(body)
            meta.append({"filename": fname, "subfolder": sub, "type": "output"})
            payload.append(base64.b64encode(body).decode("ascii"))

        return {"ui": {"ansi": meta, "ans_b64": payload}}


NODE_CLASS_MAPPINGS = {"AnsiQuantize": AnsiQuantize, "SaveAnsi": SaveAnsi}
NODE_DISPLAY_NAME_MAPPINGS = {"AnsiQuantize": "ANSI Art Quantize",
                              "SaveAnsi": "Save ANSI (.ans)"}
