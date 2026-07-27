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
import html
import json
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
 main{flex:1;display:flex;gap:14px;padding:14px;overflow:hidden}
 #stage{flex:1;display:flex;flex-direction:column;gap:8px;min-width:0}
 #img{flex:1;object-fit:contain;image-rendering:pixelated;background:#000;
      border:1px solid #282828;min-height:0}
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
  <span id="pos"></span>
  <div id="bar"><div id="fill"></div></div>
  <span id="count" class="done"></span>
  <button onclick="exportJSON()" class="primary">Export JSON</button>
</header>
<main>
  <div id="stage">
    <img id="img" alt="">
    <div class="hint">
      <kbd>Tab</kbd> next field &nbsp; <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save &amp; next &nbsp;
      <kbd>Ctrl</kbd>+<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move &nbsp; <kbd>Ctrl</kbd>+<kbd>S</kbd> skip
    </div>
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
const KEY = 'ansimon-captions-' + DATA.length;
let caps = JSON.parse(localStorage.getItem(KEY) || '{}');
let i = 0;

const $ = id => document.getElementById(id);
function render(){
  const d = DATA[i];
  $('img').src = d.img;
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
    ap.add_argument("--max-px", type=int, default=0,
                    help="downscale embedded images to this width (0 = as-is)")
    a = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(a.dataset))
    imgs = sorted(glob.glob(os.path.join(root, "*", "*.png")) +
                  glob.glob(os.path.join(root, "*.png")))
    if not imgs:
        sys.exit(f"no .png found under {root}")

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

    out = a.out or os.path.join(root, "captions.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(TEMPLATE % {"n": len(data), "data": json.dumps(data),
                            "token": json.dumps(a.token)})
    mb = os.path.getsize(out) / 1e6
    print(f"\n  {len(data)} pieces -> {out}  ({mb:.1f} MB, self-contained)")
    print(f"  open it, caption, hit Export, then:")
    print(f"    python3 tools/apply-captions.py {root} ~/Downloads/captions.json\n")


if __name__ == "__main__":
    main()
