#!/usr/bin/env python3
"""ansimon — make real ANSI art from a text prompt.

This is the brains; run it through the `ansimon` wrapper, which makes sure the
ComfyUI server is running first. It talks to ComfyUI's HTTP API, so the visual
node-graph happens behind the scenes — you just give a prompt.

The output is a PNG. The PNG is a *picture of a .ANS file* — every pixel comes
from a CP437 glyph bitmap filled with one of 16 ANSI colours — and the .ANS
that produced it is written alongside. Open that in PabloDraw, Moebius or a
terminal and you get the same image back.

Shares pixelmon's ComfyUI engine, servers.json render farm, and CLI
conventions.
"""
import argparse
import base64
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ansimon can render on a REMOTE ComfyUI (e.g. a faster box on the LAN). Choose a
# target with `--server NAME` (an alias from servers.json) or `--server host[:port]`/URL,
# or the ANSIMON_SERVER env var. Default is local. When the target is remote, results
# are fetched back over HTTP (/view) — no shared filesystem needed. Actual resolution
# happens in main(); these module-level values are the local default.
SERVER = "http://127.0.0.1:8188"
REMOTE = False
POOL = []   # >1 entry (--server a,b,c) turns on render-farm mode (jobs fan across GPUs)
COMFY = os.path.expanduser("~/ComfyUI")
OUTPUT = os.path.join(COMFY, "output")
NODE_DIR = os.path.join(COMFY, "custom_nodes", "ansi_quantize")

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Pull palette and charset names straight from the node's own registries so the
# CLI and the node can never drift apart (and so --list-palettes reflects any
# .GPL you drop into the node's gpl/ folder).
for _cand in (NODE_DIR, os.path.join(_SCRIPT_DIR, "custom_nodes", "ansi_quantize")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break
try:
    import palette as _pal
    import cp437 as _cp
    PALETTES = list(_pal.ALL_PALETTES.keys())
    CHARSETS = list(_cp.CHARSETS.keys())
except Exception:
    _pal = _cp = None
    PALETTES = ["ansi", "xterm", "vga-soft"]
    CHARSETS = ["halfblock", "blocks", "geometric", "structure", "ascii", "full"]

# Style guides (prompt snippets) loaded from styles.json next to this script.
# {name: {"prompt": "...added to positive...", "negative": "...added to negative..."}}
try:
    with open(os.path.join(_SCRIPT_DIR, "styles.json"), encoding="utf-8") as _sf:
        STYLES = {k: v for k, v in json.load(_sf).items() if not k.startswith("_")}
except Exception:
    STYLES = {}

# Named ComfyUI targets for `--server NAME`. Your personal servers.json (gitignored)
# is loaded if present; otherwise just the built-in 'local'. Copy servers.example.json
# to servers.json and add your machines, e.g. {"titan": "http://192.168.1.20:8188"}.
try:
    with open(os.path.join(_SCRIPT_DIR, "servers.json"), encoding="utf-8") as _svf:
        SERVERS = {k: v for k, v in json.load(_svf).items() if not k.startswith("_")}
except Exception:
    SERVERS = {}
SERVERS.setdefault("local", "http://127.0.0.1:8188")

# The default negative fights the two failure modes that ruin ANSI conversion:
# photographic depth (which has no flat regions for the matcher to lock onto)
# and fine detail (which turns to noise once you only have 80 columns).
DEFAULT_NEGATIVE = ("photograph, photorealistic, 3d render, depth of field, blurry, "
                    "soft focus, gradient mesh, fine detail, tiny text, watermark, "
                    "signature, jpeg artifacts, noise, grain")

# Canonical canvas presets. 8x16 cells mean cols:rows of 2:1 is a square image,
# which is why 80x40 is the default — it matches a square SDXL render exactly.
PRESETS = {
    "bbs": (80, 25),        # the classic 80x25 DOS text screen -> 640x400
    "square": (80, 40),     # square canvas -> 640x640
    "vga50": (80, 50),      # VGA 50-line mode -> 640x800
    "wide": (132, 43),      # 132-column text mode
    "tall": (80, 60),
}


def resolve_server(value):
    """Resolve a --server value (a servers.json alias, or host[:port]/full URL) to a URL."""
    import urllib.parse
    url = SERVERS.get(value, value)            # alias if known, else treat as host/URL
    if "://" not in url:
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.port:                        # default ComfyUI port
        url = f"{parsed.scheme}://{parsed.hostname}:8188"
    return url.rstrip("/")


def _colors():
    """ANSI color codes — auto-disabled when piped or NO_COLOR is set."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return {k: "" for k in ("b", "dim", "cyan", "grn", "yel", "mag", "rst")}
    return {"b": "\033[1m", "dim": "\033[2m", "cyan": "\033[36m", "grn": "\033[32m",
            "yel": "\033[33m", "mag": "\033[35m", "rst": "\033[0m"}


C = _colors()


def print_help():
    c = C

    def opt(flag, desc, default=""):
        tail = f"  {c['dim']}[{default}]{c['rst']}" if default else ""
        return f"  {c['grn']}{flag:<21}{c['rst']} {desc}{tail}"

    print(f"""
{c['b']}{c['mag']}ansimon{c['rst']} — text prompt {c['dim']}->{c['rst']} real ANSI art {c['dim']}->{c['rst']} PNG

{c['b']}USAGE{c['rst']}
  {c['cyan']}ansimon{c['rst']} "a fierce dragon"
  {c['cyan']}ansimon{c['rst']} "a castle at night" --size bbs --charset blocks
  {c['cyan']}ansimon{c['rst']} "a skull" -n 8 --fast --server rtx,titan,local

{c['b']}CANVAS{c['rst']}
{opt('--size WxH|PRESET', 'canvas in CHARACTERS, or a preset name', '80x40')}
{opt('', f"presets: {', '.join(f'{k}={v[0]}x{v[1]}' for k, v in PRESETS.items())}")}
{opt('--charset NAME', 'which CP437 characters may be used', 'blocks')}
{opt('', f"one of: {', '.join(CHARSETS)}  (--list-charsets)")}
{opt('--palette NAME', 'the 16 colours to render with', 'ansi')}
{opt('--aspect MODE', "'square' or 'classic' (4:3 CRT stretch)", 'square')}

{c['b']}LOOK{c['rst']}
{opt('--style NAMES', 'proven prompt guide(s), comma-separated (--list-styles)')}
{opt('--dither / --no-dither', 'cell-level error diffusion', 'on')}
{opt('--dither-strength N', 'damping, 0-1. lower = less colour noise', '0.75')}
{opt('--ice / --no-ice', 'iCE colours: 16 backgrounds, no blink', 'on')}
{opt('--black-bg', 'pin every background to black (BBS look)')}
{opt('--negative TEXT', 'override the default negative prompt')}

{c['b']}GENERATION{c['rst']}
{opt('-n, --number N', 'how many to make, each a different seed', '1')}
{opt('--batch "a,b,c"', 'round-robin subjects, one of each per pass')}
{opt('--seed N', 'lock / repeat a result', 'random')}
{opt('--steps N', 'sampling steps', '30')}
{opt('--cfg N', 'prompt adherence', '7.0')}
{opt('--fast', '8 steps via LCM LoRA: ~3x faster, rougher')}
{opt('--res N', 'SDXL generation resolution', '1024')}
{opt('--no-lora', 'skip the ANSI LoRA (plain SDXL)')}

{c['b']}GRID ALIGNMENT{c['rst']}
{opt('--pixel-grid N', 'flatten to an N-px logical grid first', 'auto')}
{opt('--snap-pixels', "use pixelmon's Rust pixel-snapper to lock the grid")}
{opt('--smooth MODE', 'mode | median | none', 'mode')}

{c['b']}OUTPUT{c['rst']}
{opt('--name BASE', 'output filename base', 'from prompt')}
{opt('--output-to DIR', 'save results here instead of ComfyUI/output')}
{opt('--format FMT', 'ans | xb | both | none', 'ans')}
{opt('', '.ans has no width field; .xb (XBin) does, and embeds')}
{opt('', 'the font + palette. Use xb for non-80-column canvases.')}
{opt('--xb', 'shorthand for --format xb')}
{opt('--no-sauce', 'omit the SAUCE metadata record')}
{opt('--ans-only', 'write only the art file, discard the PNG')}
{opt('--preview', 'also save the aspect-corrected preview')}
{opt('--view-scale N', 'preview zoom factor', '1')}
{opt('--title / --author / --group', 'SAUCE record fields')}
{opt('--no-open', "don't auto-open the result")}

{c['b']}FARM{c['rst']}
{opt('--server NAME[,...]', 'remote ComfyUI; comma-list = render farm', 'local')}
{opt('', f"known: {', '.join(k for k in SERVERS)}")}

{c['b']}LISTS{c['rst']}
{opt('--list-charsets', 'the CP437 subsets and what each is for')}
{opt('--list-palettes', 'the available 16-colour palettes')}
{opt('--list-styles', 'the prompt style guides')}
{opt('--doctor', 'check fonts, node install and server reachability')}
""")


def print_charsets():
    print(f"\n{C['b']}character sets{C['rst']}  {C['dim']}(--charset){C['rst']}\n")
    blurb = {
        "halfblock": "space, full block, upper half. The safest vocabulary — "
                     "always reads as real blockart, no glyph ambiguity.",
        "blocks": "half blocks + the shade ramp. Classic BBS gradients.",
        "geometric": "every block/shade/half, plus the centred square.",
        "structure": "geometric + box drawing — lets the matcher find lines.",
        "ascii": "printable 7-bit only. True ASCII art, colour via attributes.",
        "full": "the whole code page. Most detail, most noise.",
    }
    for name in CHARSETS:
        n = len(_cp.CHARSETS[name]) if _cp else "?"
        print(f"  {C['grn']}{name:<12}{C['rst']} {C['dim']}{n:>3} chars{C['rst']}  "
              f"{blurb.get(name, '')}")
    print()


def print_palettes():
    print(f"\n{C['b']}palettes{C['rst']}  {C['dim']}(--palette; 16 colours, ANSI order){C['rst']}\n")
    for name in PALETTES:
        print(f"  {C['grn']}{name}{C['rst']}")
    print(f"\n  {C['dim']}Custom{C['rst']}  with --custom-hex "
          f"\"000000,0000AA,...\" (exactly 16)\n")


def print_styles():
    if not STYLES:
        print("no styles.json found next to ansimon.py")
        return
    print(f"\n{C['b']}style guides{C['rst']}  {C['dim']}(--style a,b){C['rst']}\n")
    for name, s in STYLES.items():
        print(f"  {C['grn']}{name:<14}{C['rst']} {s.get('prompt', '')[:70]}")
        if s.get("negative"):
            print(f"  {'':<14} {C['dim']}neg: {s['negative'][:64]}{C['rst']}")
    print()


def doctor():
    print(f"\n{C['b']}ansimon doctor{C['rst']}\n")
    ok = True
    if _cp:
        print(f"  {C['grn']}font{C['rst']}      {_cp.font_source_description()}")
        b = _cp.glyph_bitmaps()
        blank = [c for c in range(256) if not b[c].any()]
        print(f"  {C['grn']}glyphs{C['rst']}    256 loaded, {len(blank)} blank "
              f"({', '.join(hex(c) for c in blank)})")
    else:
        ok = False
        print(f"  {C['yel']}font{C['rst']}      node package not importable — run ./install.sh")
    print(f"  {'✅' if os.path.isdir(NODE_DIR) else '❌'} node      {NODE_DIR}")
    ok &= os.path.isdir(NODE_DIR)
    for name, url in SERVERS.items():
        u = resolve_server(url)
        print(f"  {'✅' if server_up(u) else '  '} server    {name:<10} {u}")
    print()
    return 0 if ok else 1


def slug(text):
    out = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
    return out[:40] or "art"


def convert_existing(a):
    """`--from-ans`: read existing art and re-emit it — no GPU, no model.

    ansimon already owns a CP437 glyph table, a renderer, an XBin writer and a
    terminal-accurate .ANS reader, so converting art it didn't generate is
    almost free. Useful for turning a folder of scene files into PNGs, or
    lifting old .ANS into XBin so the width and font travel with the file.
    """
    import glob
    # These modules use relative imports, so the PACKAGE's parent has to be on
    # the path — not the package directory itself (which is what the top of
    # this file adds, for the flat `palette`/`cp437` lookups).
    for parent in (os.path.join(_SCRIPT_DIR, "custom_nodes"),
                   os.path.join(COMFY, "custom_nodes")):
        if os.path.isdir(os.path.join(parent, "ansi_quantize")):
            sys.path.insert(0, parent)
            break
    else:
        sys.exit("can't find the ansi_quantize package — run ./install.sh")
    from ansi_quantize import ansi as A, xbin as X
    from ansi_quantize.nodes import render_cells
    from ansi_quantize.palette import parse_palette
    import numpy as np
    from PIL import Image

    src = os.path.abspath(os.path.expanduser(a.from_ans))
    files = (sorted(glob.glob(os.path.join(src, "*.ans")) +
                    glob.glob(os.path.join(src, "*.xb")))
             if os.path.isdir(src) else [src])
    if not files:
        sys.exit(f"--from-ans: nothing to convert at {src}")

    pal = np.asarray(parse_palette(a.palette, a.custom_hex), np.uint8)
    dest = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to \
        else os.path.join(OUTPUT, "ansimon")
    os.makedirs(dest, exist_ok=True)

    print(f"\n{C['mag']}ansimon{C['rst']} converting {len(files)} file(s)"
          f"  {C['dim']}-> {dest}{C['rst']}")
    done = 0
    for f in files:
        raw = open(f, "rb").read()
        base = os.path.splitext(os.path.basename(f))[0]
        try:
            if raw[:5] == X.MAGIC:
                body = A.read_sauce(raw)[1]
                p = X.parse_xbin(body)
                ch, fg, bg = p["ch"], p["fg"], p["bg"]
                ice = p["ice"]
            else:
                ch, fg, bg = A.parse_ans(raw)
                ice = (A.read_sauce(raw)[0] or {}).get("ice", False)
        except Exception as e:
            print(f"   {C['yel']}skip{C['rst']} {base}: {e}")
            continue

        if a.rows:                       # --rows crops to the first N rows
            ch, fg, bg = ch[:a.rows], fg[:a.rows], bg[:a.rows]

        out = []
        if not a.ans_only:
            png = os.path.join(dest, base + ".png")
            Image.fromarray(render_cells(ch, fg, bg, pal)).save(png)
            out.append(png)
        if a.format in ("xb", "both"):
            p2 = os.path.join(dest, base + ".xb")
            X.write_xbin(p2, ch, fg, bg, palette=pal, ice=ice,
                         compress=not a.no_compress,
                         embed_font=not a.no_embed_font)
            out.append(p2)
        if a.format in ("ans", "both") and not f.endswith(".ans"):
            p3 = os.path.join(dest, base + ".ans")
            A.write_ans(p3, ch, fg, bg, ice=ice, sauce=not a.no_sauce,
                        title=a.title or base, author=a.author, group=a.group,
                        date=a.date)
            out.append(p3)
        done += 1
        print(f"   ✅ {base:<22} {ch.shape[1]}x{ch.shape[0]} cells  ->  "
              f"{', '.join(os.path.basename(o) for o in out)}")
    print(f"   {C['dim']}converted {done}/{len(files)}{C['rst']}\n")
    return 0


def parse_size(spec):
    """'80x40' | '80' | a preset name -> (cols, rows)."""
    spec = (spec or "").strip().lower()
    if spec in PRESETS:
        return PRESETS[spec]
    if "x" in spec:
        w, _, h = spec.partition("x")
        return int(w), int(h)
    if spec.isdigit():
        c = int(spec)
        return c, max(4, round(c / 2))          # 2:1 keeps the canvas square
    raise ValueError(f"bad --size {spec!r} — try 80x40, 80, or one of: "
                     f"{', '.join(PRESETS)}")


def gen_size(cols, rows, res):
    """Pick an SDXL latent size whose aspect matches the character canvas.

    A cell is 8x16, so an 80x40 canvas is square while an 80x25 one is 1.6:1
    wide. Generating square and then squashing into a wide canvas would stretch
    everything, so the latent gets the canvas's aspect, rounded to the /64 grid
    SDXL wants.
    """
    aspect = (cols * 8) / (rows * 16)
    area = res * res
    w = (area * aspect) ** 0.5
    h = w / aspect
    # Floor of 384, not 512: SDXL never lands here anyway at --res 1024, but
    # SD 1.5 legitimately trains and generates at 640x384 (an 80x24 screen at
    # 1:1), and clamping that to 512 silently changes the aspect the model was
    # taught. Anything below 384 is degenerate for either model.
    q = lambda v: max(384, int(round(v / 64)) * 64)
    return q(w), q(h)


def build_graph(a, seed, subject=None, server=None):
    subject = subject if subject is not None else a.prompt
    # The ANSI LoRA does the heavy lifting; the base prompt stays simple and
    # --style snippets steer. "flat colors, bold shapes" is what survives the
    # trip down to 80 columns and 16 colours — anything subtle does not.
    if a.raw_prompt:
        # Verbatim. A LoRA you trained yourself has its own trigger word and
        # caption shape; wrapping it in another LoRA's trigger ("ansiart") and
        # a canned tail actively fights it.
        prompt = subject
    else:
        parts = [f"ansiart, {subject}"]
        if a.style_add:
            parts.append(a.style_add)
        parts.append("bold shapes, flat colors, high contrast, simple background")
        prompt = ", ".join(parts)
    negative = a.negative + ((", " + a.style_neg) if a.style_neg else "")

    name = slug(subject) if a.batch else (a.name or slug(subject))
    # seed in the filename so each variation is identifiable and re-runnable.
    prefix = f"ansimon/{name}_{a.cols}x{a.rows}_{a.charset}_s{seed}"

    g = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.base}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": a.gen_w, "height": a.gen_h, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": None, "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": None, "text": negative}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": a.steps, "cfg": a.cfg,
                         "sampler_name": a.sampler, "scheduler": a.scheduler,
                         "denoise": 1.0, "model": None, "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "10": {"class_type": "AnsiQuantize",
               "inputs": {"image": ["8", 0], "cols": a.cols, "rows": a.rows,
                          "charset": a.charset, "palette": a.palette,
                          "ice_colors": a.ice, "dither": a.dither,
                          "dither_strength": a.dither_strength,
                          "smooth": a.smooth, "pixel_grid": a.pixel_grid,
                          "snap_pixels": a.snap_pixels, "snap_colors": a.snap_colors,
                          "aspect": ("classic (4:3)" if a.aspect.startswith("c")
                                     else "square"),
                          "view_scale": a.view_scale, "resample": a.filter,
                          "force_black_bg": a.black_bg, "custom_hex": a.custom_hex}},
    }
    if not a.ans_only:
        g["11"] = {"class_type": "SaveImage",
                   "inputs": {"filename_prefix": prefix + "_ansi", "images": ["10", 0]}}
    if a.preview:
        g["12"] = {"class_type": "SaveImage",
                   "inputs": {"filename_prefix": prefix + "_preview", "images": ["10", 1]}}
    if a.format != "none":
        g["13"] = {"class_type": "SaveAnsi",
                   "inputs": {"cells_b64": ["10", 2], "filename_prefix": prefix,
                              "format": a.format,
                              "title": a.title or subject[:35], "author": a.author,
                              "group": a.group, "date": a.date,
                              "sauce": not a.no_sauce,
                              "xb_compress": not a.no_compress,
                              "xb_embed_font": not a.no_embed_font}}

    # Chain LoRAs onto the base: SDXL -> [ANSI Art XL] -> [LCM if --fast].
    # Each LoraLoader patches both the model and the text encoder (clip), so we
    # thread the "current" source through and wire the sampler/prompts to the end.
    model_src, clip_src = ["4", 0], ["4", 1]
    if not a.no_lora:
        g["15"] = {"class_type": "LoraLoader",
                   "inputs": {"model": model_src, "clip": clip_src,
                              "lora_name": a.lora,
                              "strength_model": a.lora_strength,
                              "strength_clip": a.lora_strength}}
        model_src, clip_src = ["15", 0], ["15", 1]
    if a.fast:
        g["16"] = {"class_type": "LoraLoader",
                   "inputs": {"model": model_src, "clip": clip_src,
                              "lora_name": a.lcm_lora,
                              "strength_model": 1.0, "strength_clip": 1.0}}
        model_src, clip_src = ["16", 0], ["16", 1]

    g["6"]["inputs"]["clip"] = clip_src
    g["7"]["inputs"]["clip"] = clip_src
    g["3"]["inputs"]["model"] = model_src
    return g


class SubmitError(Exception):
    """ComfyUI refused the graph. Carries the reason so the farm can report it."""


def submit(graph, server=None, raising=False):
    """POST a graph to ComfyUI -> prompt_id.

    `raising=True` turns rejections into SubmitError instead of exiting, which
    is what the farm wants: one box missing the ansi_quantize node should cost
    you that box, not the whole run.
    """
    server = server or SERVER
    data = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(f"{server}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:1200]
        msg = "ComfyUI rejected the request:\n" + body
        if "AnsiQuantize" in body or "SaveAnsi" in body:
            msg += ("\n\nThat box doesn't have the ansi_quantize node. Install it there:\n"
                    "    git clone <your-remote> ~/git/ansimon && ~/git/ansimon/install.sh\n"
                    "then restart its ComfyUI.")
        if raising:
            raise SubmitError(msg)
        sys.exit(msg)
    except urllib.error.URLError as e:
        msg = f"Couldn't reach ComfyUI at {server} — is the server running? ({e})"
        if raising:
            raise SubmitError(msg)
        sys.exit(msg)


def wait(pid, server=None, timeout=900):
    server = server or SERVER
    for _ in range(timeout):
        with urllib.request.urlopen(f"{server}/history/{pid}", timeout=30) as r:
            hist = json.loads(r.read())
        if pid in hist and hist[pid].get("outputs"):
            return hist[pid]["outputs"]
        time.sleep(1)
    sys.exit("Timed out waiting for the art.")


def poll(pid, server):
    """One non-blocking /history check; returns the outputs dict, or None if not ready."""
    with urllib.request.urlopen(f"{server}/history/{pid}", timeout=30) as r:
        hist = json.loads(r.read())
    if pid in hist and hist[pid].get("outputs"):
        return hist[pid]["outputs"]
    return None


def server_up(server):
    try:
        urllib.request.urlopen(f"{server}/system_stats", timeout=5).read()
        return True
    except Exception:
        return False


def _short(url):
    return url.split("//", 1)[-1]


def fetch_image(im, dest_dir, server=None):
    """Download one server-side output image via /view into dest_dir; return local path."""
    import urllib.parse
    server = server or SERVER
    q = urllib.parse.urlencode({"filename": im["filename"],
                                "subfolder": im.get("subfolder", ""),
                                "type": im.get("type", "output")})
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, im["filename"])
    with urllib.request.urlopen(f"{server}/view?{q}", timeout=180) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)
    return out


def save_ans(outs, dest_dir):
    """Pull the .ANS out of the SaveAnsi node's UI payload and write it locally.

    The bytes travel inline in /history rather than as a second /view fetch,
    which is what lets a farm node hand back a complete result without us
    needing any access to its filesystem.
    """
    written = []
    for node in outs.values():
        blobs = node.get("ans_b64") or []
        metas = node.get("ansi") or []
        for i, b64 in enumerate(blobs):
            meta = metas[i] if i < len(metas) else {}
            fname = meta.get("filename") or f"art_{i:05d}.ans"
            os.makedirs(dest_dir, exist_ok=True)
            path = os.path.join(dest_dir, fname)
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            written.append(path)
    return written


def collect(outs, dest_dir, server):
    """Fetch every artefact for one finished job -> (png_paths, ans_paths)."""
    imgs = [im for node in outs.values() for im in node.get("images", [])]
    pngs = [fetch_image(im, dest_dir, server) for im in imgs]
    return pngs, save_ans(outs, dest_dir)


def run_farm(a, work):
    """Render farm: distribute jobs across POOL with dynamic dispatch (feed the free GPU).
    Faster boxes naturally pull more jobs; results are fetched back from whichever GPU made them."""
    live = [s for s in POOL if server_up(s)]
    down = [s for s in POOL if s not in live]
    if down:
        print(f"   ⚠ skipping unreachable: {', '.join(_short(s) for s in down)}")
    if not live:
        sys.exit("render farm: no reachable servers in the pool.")
    print(f"   \U0001f69c render farm: {len(live)} GPU(s) — {', '.join(_short(s) for s in live)}")
    pending = list(work)        # (subject, seed, dest)
    inflight = {}               # server -> (subject, seed, dest, pid)
    total = len(work)
    done = 0

    def launch(srv):
        """Submit the next pending job to srv. False = server unusable (drop it)."""
        while pending:
            subj, seed, d = pending.pop(0)
            try:
                pid = submit(build_graph(a, seed, subject=subj, server=srv), srv,
                             raising=True)
            except SubmitError as e:
                pending.insert(0, (subj, seed, d))   # couldn't submit; keep the job
                # Say WHY the box is being dropped — a silently shrinking farm
                # looks like the job just ran slowly on one GPU.
                first = str(e).strip().splitlines()
                hint = "missing the ansi_quantize node" if "ansi_quantize" in str(e) \
                    else (first[-1] if first else "submit failed")
                print(f"   ⚠ dropping {_short(srv)}: {hint}")
                return False
            inflight[srv] = (subj, seed, d, pid)
            return True
        return True             # nothing left to do

    for srv in list(live):
        if launch(srv) is False:
            live.remove(srv)
    if not live and not inflight:
        sys.exit("render farm: every server refused the job (see above).")

    while inflight:
        advanced = False
        for srv, (subj, seed, d, pid) in list(inflight.items()):
            try:
                outs = poll(pid, srv)
            except Exception:
                print(f"   ⚠ {_short(srv)} unreachable — requeueing its job")
                pending.append((subj, seed, d))
                del inflight[srv]
                advanced = True
                continue
            if outs is None:
                continue
            advanced = True
            dest_dir = d or os.path.join(OUTPUT, "ansimon")
            pngs, anses = collect(outs, dest_dir, srv)
            done += 1
            shown = next((f for f in pngs if "_ansi_" in f), None) or \
                    (anses[0] if anses else None)
            sj = f"{subj}  " if a.batch else ""
            print(f"   ✅ [{done}/{total}] {_short(srv):<20} {sj}seed={seed}  ->  {shown}")
            del inflight[srv]
            launch(srv)          # feed the now-free GPU its next job
        if not advanced:
            time.sleep(1)

    if pending:
        print(f"   ⚠ {len(pending)} job(s) left undone (all GPUs dropped).")


def open_file(path):
    for cmd in ("xdg-open", "open"):
        if shutil.which(cmd):
            subprocess.Popen([cmd, path], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return


def main():
    global SERVER, REMOTE, POOL

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-h", "--help", action="store_true", dest="show_help")
    p.add_argument("prompt", nargs="?", help='what to draw, e.g. "a fierce dragon"')
    p.add_argument("--size", default="80x40",
                   help="canvas in CHARACTERS: WxH, W, or a preset name. default 80x40")
    p.add_argument("--cols", type=int, default=None, help="canvas width in characters")
    p.add_argument("--rows", type=int, default=None, help="canvas height in characters")
    p.add_argument("-n", "--number", type=int, default=1,
                   help="how many to make, each a different seed. default 1")
    p.add_argument("--batch", default=None, metavar="SUBJECTS",
                   help="comma-separated subjects; one of each per pass")
    p.add_argument("--charset", default="blocks",
                   help=f"CP437 subset: {', '.join(CHARSETS)}. default blocks")
    p.add_argument("--palette", default="ansi",
                   help=f"16-colour palette: {', '.join(PALETTES)}. default ansi")
    p.add_argument("--aspect", default="square",
                   help="'square' or 'classic' (4:3 CRT stretch). default square")
    p.add_argument("--style", default="",
                   help="comma-separated style guides (--list-styles)")
    p.add_argument("--dither", action="store_true", default=True,
                   help="cell-level error diffusion (default on)")
    p.add_argument("--no-dither", dest="dither", action="store_false",
                   help="disable error diffusion — flatter, blockier")
    p.add_argument("--dither-strength", dest="dither_strength", type=float,
                   default=0.75, help="0-1 damping. lower = less colour noise")
    p.add_argument("--ice", action="store_true", default=True,
                   help="iCE colours: 16 backgrounds, no blink (default on)")
    p.add_argument("--no-ice", dest="ice", action="store_false",
                   help="classic 8 backgrounds + blink bit")
    p.add_argument("--black-bg", dest="black_bg", action="store_true",
                   help="pin every background to black")
    p.add_argument("--steps", type=int, default=None, help="sampling steps. default 30")
    p.add_argument("--cfg", type=float, default=None, help="prompt adherence. default 7.0")
    p.add_argument("--seed", type=int, default=-1, help="-1 = random each run")
    p.add_argument("--raw-prompt", dest="raw_prompt", action="store_true",
                   help="use the prompt verbatim — no 'ansiart' prefix or canned tail")
    p.add_argument("--negative", default=DEFAULT_NEGATIVE,
                   help="override the default negative prompt")
    p.add_argument("--name", default=None, help="output filename base (default: from prompt)")
    p.add_argument("--output-to", dest="output_to", default=None, metavar="DIR",
                   help="save results here instead of ComfyUI/output")
    p.add_argument("--no-subdirs", dest="no_subdirs", action="store_true",
                   help="don't create per-subject subfolders in --batch mode")
    p.add_argument("--custom-hex", dest="custom_hex", default="",
                   help="16 hex colours for --palette Custom")
    p.add_argument("--preview", action="store_true",
                   help="also save the aspect-corrected preview image")
    p.add_argument("--view-scale", dest="view_scale", type=int, default=1,
                   help="preview zoom factor. default 1")
    p.add_argument("--format", default="ans", choices=["ans", "xb", "both", "none"],
                   help="art file format alongside the PNG. default ans")
    p.add_argument("--xb", dest="want_xb", action="store_true",
                   help="shorthand for --format xb")
    p.add_argument("--no-ans", dest="no_ans", action="store_true",
                   help="don't write any art sidecar (same as --format none)")
    p.add_argument("--ans-only", dest="ans_only", action="store_true",
                   help="write only the art file, discard the PNG")
    p.add_argument("--no-sauce", dest="no_sauce", action="store_true",
                   help="omit the SAUCE metadata record")
    p.add_argument("--no-compress", dest="no_compress", action="store_true",
                   help="store XBin uncompressed")
    p.add_argument("--no-embed-font", dest="no_embed_font", action="store_true",
                   help="don't embed the CP437 font in the XBin")
    p.add_argument("--title", default="", help="SAUCE title")
    p.add_argument("--author", default="", help="SAUCE author")
    p.add_argument("--group", default="", help="SAUCE group")
    p.add_argument("--date", default="", help="SAUCE date YYYYMMDD (default: today)")
    p.add_argument("--server", default=None, metavar="NAME|HOST[,...]",
                   help="remote ComfyUI; comma-list = render farm")
    p.add_argument("--base", default="sd_xl_base_1.0.safetensors",
                   help="SDXL base checkpoint")
    p.add_argument("--lora", default="ansi-art-xl.safetensors", help="ANSI-art LoRA")
    p.add_argument("--lora-strength", dest="lora_strength", type=float, default=0.9,
                   help="LoRA weight. default 0.9")
    p.add_argument("--no-lora", dest="no_lora", action="store_true",
                   help="skip the ANSI LoRA (plain SDXL)")
    p.add_argument("--lcm-lora", dest="lcm_lora", default="lcm-lora-sdxl.safetensors",
                   help="LCM LoRA used by --fast")
    p.add_argument("--fast", action="store_true", help="8 steps via LCM: ~3x faster")
    p.add_argument("--res", type=int, default=1024,
                   help="SDXL generation resolution. default 1024")
    p.add_argument("--pixel-grid", dest="pixel_grid", type=int, default=0,
                   help="flatten to an N-px logical grid before the character grid")
    p.add_argument("--snap-pixels", dest="snap_pixels", action="store_true",
                   help="use pixelmon's Rust pixel-snapper to lock the grid")
    p.add_argument("--snap-colors", dest="snap_colors", type=int, default=0,
                   help="colour cap for --snap-pixels. default auto")
    p.add_argument("--smooth", choices=["mode", "median", "none"], default="mode",
                   help="grid-recovery filter. default mode")
    p.add_argument("--filter", choices=list(("nearest", "box (area average)", "lanczos")),
                   default="box (area average)", help="cell sampling filter")
    p.add_argument("--sampler", default="dpmpp_2m", help="ksampler sampler_name")
    p.add_argument("--scheduler", default="karras", help="ksampler scheduler")
    p.add_argument("--list-charsets", action="store_true", help="list charsets and exit")
    p.add_argument("--list-palettes", action="store_true", help="list palettes and exit")
    p.add_argument("--list-styles", action="store_true", help="list style guides and exit")
    p.add_argument("--doctor", action="store_true", help="check the install and exit")
    p.add_argument("--from-ans", dest="from_ans", default=None, metavar="FILE|DIR",
                   help="skip generation: read existing .ans/.xb and convert")
    p.add_argument("--no-open", action="store_true", help="don't auto-open the result")

    a = p.parse_args()

    if a.show_help or (not a.prompt and not a.batch and not any(
            (a.list_charsets, a.list_palettes, a.list_styles, a.doctor,
             a.from_ans))):
        print_help()
        return 0
    if a.list_charsets:
        print_charsets()
        return 0
    if a.list_palettes:
        print_palettes()
        return 0
    if a.list_styles:
        print_styles()
        return 0
    if a.doctor:
        return doctor()
    if a.from_ans:
        return convert_existing(a)

    # --- canvas -----------------------------------------------------------
    try:
        cols, rows = parse_size(a.size)
    except ValueError as e:
        sys.exit(str(e))
    a.cols = a.cols or cols
    a.rows = a.rows or rows
    a.gen_w, a.gen_h = gen_size(a.cols, a.rows, a.res)

    # --format is the real control; --xb / --no-ans are conveniences on top.
    if a.want_xb and a.format == "ans":
        a.format = "xb"
    if a.no_ans:
        a.format = "none"
    if a.format == "none" and a.ans_only:
        sys.exit("--ans-only with no art format would write nothing at all.")
    # .ANS carries no width field, so a non-80-column canvas is only reliably
    # reopenable as XBin. Say so once rather than let it silently mangle later.
    if a.cols != 80 and a.format == "ans":
        print(f"   {C['yel']}note{C['rst']} {a.cols}-column .ans depends on the "
              f"viewer reading SAUCE for width; --format xb puts it in the header")

    # --- defaults that depend on --fast -----------------------------------
    if a.steps is None:
        a.steps = 8 if a.fast else 30
    if a.cfg is None:
        a.cfg = 1.8 if a.fast else 7.0
    if a.fast and a.sampler == "dpmpp_2m":
        a.sampler, a.scheduler = "lcm", "normal"
    if not a.date:
        a.date = time.strftime("%Y%m%d")

    # --- styles -----------------------------------------------------------
    a.style_add, a.style_neg = "", ""
    if a.style:
        adds, negs = [], []
        for nm in [s.strip() for s in a.style.split(",") if s.strip()]:
            if nm not in STYLES:
                sys.exit(f"unknown style {nm!r} — see --list-styles")
            adds.append(STYLES[nm].get("prompt", ""))
            if STYLES[nm].get("negative"):
                negs.append(STYLES[nm]["negative"])
        a.style_add = ", ".join(x for x in adds if x)
        a.style_neg = ", ".join(x for x in negs if x)

    # --- server / farm ----------------------------------------------------
    target = a.server or os.environ.get("ANSIMON_SERVER")
    if target:
        names = [t.strip() for t in target.split(",") if t.strip()]
        POOL = [resolve_server(t) for t in names]
        SERVER = POOL[0]
        REMOTE = not SERVER.startswith("http://127.0.0.1")
    if len(POOL) > 1:
        seen, uniq = set(), []
        for s in POOL:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        POOL = uniq

    # --- build the work list ----------------------------------------------
    subjects = ([s.strip() for s in a.batch.split(",") if s.strip()]
                if a.batch else [a.prompt])
    base_dir = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to \
        else os.path.join(OUTPUT, "ansimon")

    work = []
    for i in range(a.number):
        for subj in subjects:
            seed = a.seed if a.seed >= 0 else random.randint(0, 2 ** 31 - 1)
            if a.seed >= 0:
                seed = a.seed + i
            d = base_dir
            if a.batch and not a.no_subdirs:
                d = os.path.join(base_dir, slug(subj))
            work.append((subj, seed, d))

    label = a.batch or a.prompt
    print(f"\n{C['mag']}ansimon{C['rst']} {C['b']}{label}{C['rst']}")
    print(f"   {C['dim']}canvas {a.cols}x{a.rows} chars -> {a.cols*8}x{a.rows*16} px"
          f"  ·  latent {a.gen_w}x{a.gen_h}  ·  charset {a.charset}"
          f"  ·  palette {a.palette}{C['rst']}")

    t0 = time.time()
    if len(POOL) > 1:
        run_farm(a, work)
    else:
        for n, (subj, seed, d) in enumerate(work, 1):
            pid = submit(build_graph(a, seed, subject=subj), SERVER)
            outs = wait(pid, SERVER)
            pngs, anses = collect(outs, d, SERVER)
            shown = next((f for f in pngs if "_ansi_" in f), None) or \
                    (anses[0] if anses else None)
            sj = f"{subj}  " if a.batch else ""
            print(f"   ✅ [{n}/{len(work)}] {sj}seed={seed}  ->  {shown}")
            for ans in anses:
                print(f"      {C['dim']}{ans}{C['rst']}")
            if n == 1 and shown and not a.no_open:
                open_file(shown)

    print(f"   {C['dim']}all done in {time.time() - t0:.1f}s{C['rst']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
