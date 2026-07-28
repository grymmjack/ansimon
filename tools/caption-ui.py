#!/usr/bin/env python3
"""Build a self-contained captioning page for a LoRA dataset.

    python3 tools/caption-ui.py DATASET_DIR [-o captions.html]
    # caption in the browser, hit Export, then:
    python3 tools/apply-captions.py DATASET_DIR captions.json

Why this exists
---------------
The first grymmjack LoRA learned layouts instead of steerable style, and the
cause was in the captions: the "subject" slot held BBS *names* — "colly 10/96",
"borgasm electronic magazine" — never a description of what is depicted. The
model was never once shown the word "skull" attached to a skull, so at high
strength it reproduced compositions and ignored the prompt entirely.

Auto-captioning does not fix this. BLIP-large, tested on this exact corpus,
returned "a photo of a city at night" for a BORGASM graffiti logo and "a man
with a skateboard" for FOKUS. General vision models are trained on photographs
and cannot read 16-colour blockart; their output is not merely useless here, it
is actively harmful as training signal. So the human stays in the loop — but the
loop is made fast.

What you actually type is the IMAGERY, because everything else is already known:
the lettering is usually in the SAUCE title, and the piece type was classified
when the dataset was built. Both are pre-filled.

Everything is embedded in one HTML file — images as data URIs, no server, no
network. Work is saved to localStorage as you go, so closing the tab is safe.
"""
import argparse
import base64
import glob
import hashlib
import html
import json
import re
import os
import sys

TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>ansimon — caption %(n)d pieces</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
      background:#111;color:#ddd;display:flex;flex-direction:column;height:100vh}
 header{padding:8px 14px;background:#181818;border-bottom:1px solid #282828;
        display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #bar{flex:1;height:6px;background:#282828;border-radius:3px;overflow:hidden;min-width:120px}
 #fill{height:100%%;background:#4a9;width:0}
 main{flex:1;display:flex;gap:14px;padding:14px;overflow:hidden;min-height:0}
 #stage{flex:1;display:flex;flex-direction:column;gap:8px;min-width:0;min-height:0}
 /* The image must never dictate layout height. A 24-row screen is 640x384 but
    a full piece can be many screens tall, so the viewport scrolls and the img
    is clamped to it — otherwise the art runs off the bottom of the page with
    no way to reach it. */
 #viewport{flex:1;min-height:0;overflow:auto;background:#000;box-sizing:border-box;
           border:1px solid #282828;padding:4px;text-align:center;
           scrollbar-width:auto;scrollbar-color:#4a9 #1a1a1a}
 #viewport::-webkit-scrollbar{width:14px;height:14px}
 #viewport::-webkit-scrollbar-track{background:#1a1a1a}
 #viewport::-webkit-scrollbar-thumb{background:#4a9;border-radius:7px}
 #viewport.cropped{border-color:#c84}
 /* Fit uses width+height 100%% with object-fit, NOT max-width/max-height.
    max-* only ever scales DOWN, so a 640px-wide piece in a 1700px viewport
    would render at natural size — a postage stamp in a big black box. */
 #img{image-rendering:pixelated;vertical-align:top}
 #img.fit{display:block;width:100%%;height:100%%;object-fit:contain}
 #img.zoom{display:inline-block;height:auto}
 #zoomrow{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
 #zoomrow button{padding:5px 13px;font-size:13px}
 #zoomrow button.on{background:#2e6f5e;border-color:#6ecfb0;color:#fff;font-weight:700}
 #clip{color:#eb9;font-weight:700}
 aside{width:340px;display:flex;flex-direction:column;gap:10px}
 label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#888}
 input,textarea,select{width:100%%;background:#1c1c1c;color:#eee;border:1px solid #333;
        border-radius:4px;padding:7px;font:inherit;box-sizing:border-box}
 textarea{resize:vertical;min-height:64px}
 .hint{color:#777;font-size:12px}
 .known{background:#161616;border:1px solid #262626;border-radius:4px;padding:8px;font-size:12px}
 .known b{color:#9c9}
 kbd{background:#252525;border:1px solid #3a3a3a;border-radius:3px;padding:1px 5px;font-size:11px}
 button{background:#2a2a2a;color:#eee;border:1px solid #3a3a3a;border-radius:4px;
        padding:7px 12px;cursor:pointer;font:inherit}
 button:hover{background:#333}
 button.primary{background:#2e6f5e;border-color:#3a8a74}
 .row{display:flex;gap:8px}
 .done{color:#4a9}
</style>
<header>
  <b style="color:#c9c">ansimon</b>
  <span class="hint" title="rebuild the page if this looks stale">b%(build)s</span>
  <span style="color:#9c9;font-weight:700">%(scope_label)s</span>
  <span id="pos"></span>
  <div id="bar"><div id="fill"></div></div>
  <span id="count" class="done"></span>
  <button onclick="exportJSON()" class="primary">Export JSON</button>
</header>
<main>
  <div id="stage">
    <div id="zoomrow">
      <span class="hint">zoom</span>
      <button data-z="fit" class="on" onclick="setZoom('fit')">Fit</button>
      <button data-z="1" onclick="setZoom(1)">1&times;</button>
      <button data-z="2" onclick="setZoom(2)">2&times;</button>
      <button data-z="3" onclick="setZoom(3)">3&times;</button>
      <span class="hint" id="dims"></span>
      <span id="clip"></span>
      <span class="hint" style="margin-left:auto">
        <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save &amp; next &nbsp;
        <kbd>Ctrl</kbd>+<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move &nbsp;
        <kbd>Ctrl</kbd>+<kbd>S</kbd> skip &nbsp; <kbd>Ctrl</kbd>+<kbd>0..3</kbd> zoom
      </span>
    </div>
    <div id="viewport"><img id="img" class="fit" alt=""></div>
  </div>
  <aside>
    <div class="known" id="known"></div>
    <div>
      <label for="subject">Imagery — what is actually drawn</label>
      <textarea id="subject" placeholder="skull with wings, chrome lettering, a wizard"></textarea>
    </div>
    <div>
      <label for="kind">Piece type</label>
      <select id="kind">
        <option>blockart logo</option><option>bbs menu screen</option>
        <option>screen with text</option><option>ansi art piece</option>
        <option>font / charset</option><option>illustration</option>
      </select>
    </div>
    <div>
      <label for="extra">Extra tags (optional)</label>
      <input id="extra" placeholder="dark, chrome, gothic, symmetrical">
    </div>
    <div class="row">
      <button onclick="prev()">&larr; Prev</button>
      <button onclick="skip()">Skip</button>
      <button onclick="next()" class="primary" style="flex:1">Save &amp; Next &rarr;</button>
    </div>
    <div class="hint" id="preview" style="border-top:1px solid #282828;padding-top:8px"></div>
  </aside>
</main>
<script>
const DATA = %(data)s;
const TOKEN = %(token)s;
const SCOPE = %(scope)s;
const KEY = 'ansimon-captions-' + SCOPE + '-' + DATA.length;
let caps = JSON.parse(localStorage.getItem(KEY) || '{}');
let i = 0;
// Zoom deliberately does NOT persist. It used to, keyed on the piece count —
// which is identical between rebuilds, so a zoomed-in view survived every new
// build and was indistinguishable from a broken page. Always start at Fit.
let zoom = 'fit';

const $ = id => document.getElementById(id);
function setZoom(z){
  zoom = z;
  const img = $('img');
  document.querySelectorAll('#zoomrow button').forEach(b =>
    b.classList.toggle('on', b.dataset.z == String(z)));
  if (z === 'fit'){
    img.classList.add('fit'); img.classList.remove('zoom'); img.style.width = '';
  } else {
    img.classList.remove('fit'); img.classList.add('zoom');
    img.style.width = (img.naturalWidth * z) + 'px';
  }
  // An overflowing viewport looks exactly like a broken one, so say so.
  requestAnimationFrame(() => {
    const v = $('viewport');
    const cut = v.scrollHeight > v.clientHeight + 2 || v.scrollWidth > v.clientWidth + 2;
    v.classList.toggle('cropped', cut);
    $('clip').textContent = cut ? '\u26a0 zoomed in \u2014 scroll, or hit Fit' : '';
  });
}
function render(){
  const d = DATA[i];
  const img = $('img');
  img.onload = () => {
    $('dims').textContent = img.naturalWidth + '\u00d7' + img.naturalHeight +
      '  (' + Math.round(img.naturalWidth/8) + '\u00d7' + Math.round(img.naturalHeight/16) + ' cells)';
    setZoom(zoom);
    $('viewport').scrollTop = 0;
  };
  img.src = d.img;
  $('pos').textContent = (i+1) + ' / ' + DATA.length + '  ' + d.name;
  $('known').innerHTML = '<b>file</b> ' + d.name +
     (d.title ? '<br><b>SAUCE title</b> ' + d.title : '') +
     (d.group ? '<br><b>group</b> ' + d.group : '');
  const c = caps[d.name] || {};
  $('subject').value = c.subject !== undefined ? c.subject : '';
  $('kind').value = c.kind || d.kind;
  $('extra').value = c.extra || '';
  const n = Object.keys(caps).length;
  $('fill').style.width = (100*n/DATA.length) + '%%';
  $('count').textContent = n + ' captioned';
  updatePreview();
  $('subject').focus();
}
function caption(){
  const bits = [TOKEN, 'ansi art', $('kind').value];
  const s = $('subject').value.trim(); if (s) bits.push(s);
  const e = $('extra').value.trim();   if (e) bits.push(e);
  bits.push('cp437 block characters', '16 color ega palette', 'text mode art');
  return bits.join(', ');
}
function updatePreview(){ $('preview').textContent = caption(); }
['subject','extra','kind'].forEach(id => $(id).addEventListener('input', updatePreview));

function save(){
  caps[DATA[i].name] = {subject:$('subject').value.trim(), kind:$('kind').value,
                        extra:$('extra').value.trim(), caption:caption()};
  localStorage.setItem(KEY, JSON.stringify(caps));
}
function next(){ save(); if (i < DATA.length-1) i++; render(); }
function prev(){ save(); if (i > 0) i--; render(); }
function skip(){ if (i < DATA.length-1) i++; render(); }

document.addEventListener('keydown', ev => {
  if (ev.ctrlKey && ev.key === 'Enter'){ ev.preventDefault(); next(); }
  else if (ev.ctrlKey && ev.key === 'ArrowRight'){ ev.preventDefault(); next(); }
  else if (ev.ctrlKey && ev.key === 'ArrowLeft'){ ev.preventDefault(); prev(); }
  else if (ev.ctrlKey && ev.key.toLowerCase() === 's'){ ev.preventDefault(); skip(); }
  else if (ev.ctrlKey && '0123'.includes(ev.key)){
    ev.preventDefault(); setZoom(ev.key === '0' ? 'fit' : Number(ev.key));
  }
});

function exportJSON(){
  save();
  const out = {};
  for (const k in caps) out[k] = caps[k].caption;
  const blob = new Blob([JSON.stringify(out, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'captions.json'; a.click();
}
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset dir built by build-lora-dataset.py")
    ap.add_argument("-o", "--out", default=None, help="output .html")
    ap.add_argument("--token", default="grymmjack")
    ap.add_argument("--only", default=None, metavar="SUBSTR",
                    help="only include folders matching this, e.g. grymmjack. "
                         "A multi-artist dataset is sorted by folder, so without "
                         "this your own work can sit hundreds of pieces deep.")
    ap.add_argument("--max-px", type=int, default=0,
                    help="downscale embedded images to this width (0 = as-is)")
    a = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(a.dataset))
    imgs = sorted(glob.glob(os.path.join(root, "*", "*.png")) +
                  glob.glob(os.path.join(root, "*.png")))
    if a.only:
        want = a.only.lower()
        imgs = [p for p in imgs
                if want in os.path.basename(os.path.dirname(p)).lower()]
    if not imgs:
        sys.exit(f"no .png found under {root}" +
                 (f" matching --only {a.only!r}" if a.only else ""))

    # The manifest carries the SAUCE title and the auto-classified kind, so the
    # page can pre-fill everything already known and ask only for the imagery.
    meta = {}
    mpath = os.path.join(root, "manifest.json")
    if os.path.exists(mpath):
        for r in json.load(open(mpath, encoding="utf-8")):
            parts = r["caption"].split(", ")
            meta[r["file"]] = {"kind": parts[2] if len(parts) > 2 else "blockart logo",
                               "title": parts[3] if len(parts) > 7 else "",
                               "source": r.get("source", "")}

    data = []
    for p in imgs:
        name = os.path.basename(p)
        m = meta.get(name, {})
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data.append({"name": name, "img": "data:image/png;base64," + b64,
                     "kind": m.get("kind", "blockart logo"),
                     "title": html.escape(m.get("title", "")),
                     "group": html.escape(m.get("source", ""))})

    # The filename carries the build hash. Browsers cache file:// URLs hard —
    # a 20 MB local page can survive Ctrl+Shift+R and even a devtools
    # cache-disable — and a stale page is indistinguishable from a broken one.
    # A new build is therefore a new URL, which no cache can get wrong.
    build = hashlib.sha1((str(len(data)) + (a.only or "") + TEMPLATE)
                         .encode()).hexdigest()[:6]
    tag = re.sub(r"[^a-z0-9]+", "-", (a.only or "all").lower()).strip("-")
    out = a.out or os.path.join(root, f"captions-{tag}-{build}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(TEMPLATE % {"n": len(data), "data": json.dumps(data),
                            "token": json.dumps(a.token),
                            "scope": json.dumps(a.only or "all"),
                            "scope_label": html.escape(a.only or "all artists"),
                            "build": build})
    for stale in glob.glob(os.path.join(root, f"captions-{tag}-*.html")):
        if os.path.abspath(stale) != os.path.abspath(out):
            os.remove(stale)                      # only ever one build present
    mb = os.path.getsize(out) / 1e6
    print(f"\n  {len(data)} pieces -> {out}  ({mb:.1f} MB, self-contained)")
    print(f"  open:  file://{out}")
    print(f"  open it, caption, hit Export, then:")
    print(f"    python3 tools/apply-captions.py {root} ~/Downloads/captions.json\n")


if __name__ == "__main__":
    main()
