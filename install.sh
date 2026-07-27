#!/usr/bin/env bash
# Reproducible setup for ansimon. Idempotent — safe to re-run.
# Does NOT download models (run ./download-models.sh after).
#
# ansimon deliberately SHARES ComfyUI with pixelmon and soundmon: same engine,
# same venv, same launch script, same port, same servers.json render farm. It
# only adds its own node. So if you already run either of those on this box (or
# on a farm box), this is nearly a no-op — link the files, fetch the LoRA, done.
#
# Force a vendor with  ANSIMON_GPU=nvidia|amd|cpu ./install.sh  if needed.
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
COMFY="${COMFYUI_DIR:-$HOME/ComfyUI}"

detect_gpu() {
    case "${ANSIMON_GPU:-}" in nvidia|amd|cpu|mps) echo "$ANSIMON_GPU"; return;; esac
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        echo mps
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo nvidia
    elif [ -x /usr/lib/wsl/lib/nvidia-smi ] && /usr/lib/wsl/lib/nvidia-smi -L >/dev/null 2>&1; then
        echo nvidia
    elif [ -e /dev/kfd ] || command -v rocminfo >/dev/null 2>&1; then
        echo amd
    else
        echo cpu
    fi
}
GPU="$(detect_gpu)"

echo "==> repo:    $REPO"
echo "==> ComfyUI: $COMFY"
echo "==> GPU:     $GPU"

# 1. Engine. If pixelmon or soundmon already set ComfyUI up, reuse it wholesale.
if [ -x "$COMFY/.venv/bin/python" ]; then
    echo "==> found an existing ComfyUI + venv — reusing it (nothing to build)"
else
    echo "==> no ComfyUI here yet."
    if [ -x "$HOME/pixelmon/install.sh" ]; then
        echo "    pixelmon's install.sh is present and builds exactly the engine ansimon"
        echo "    needs (ComfyUI + venv + the right torch). Run that first:"
        echo "        ~/pixelmon/install.sh"
    else
        echo "    Clone and build ComfyUI first — the quickest path is pixelmon's installer:"
        echo "        git clone https://github.com/grymmjack/pixelmon.git ~/pixelmon"
        echo "        ~/pixelmon/install.sh"
    fi
    echo "    Then re-run this script."
    exit 1
fi

# 2. Link our files into place (this repo stays the source of truth)
link() { ln -sfn "$1" "$2"; echo "   linked $2 -> $1"; }
mkdir -p "$HOME/.local/bin" "$COMFY/custom_nodes"
chmod +x "$REPO/bin/ansimon" "$REPO/download-models.sh"
link "$REPO/ansimon.py"                  "$COMFY/ansimon.py"
link "$REPO/styles.json"                 "$COMFY/styles.json"
link "$REPO/custom_nodes/ansi_quantize"  "$COMFY/custom_nodes/ansi_quantize"
link "$REPO/bin/ansimon"                 "$HOME/.local/bin/ansimon"

# 3. Reuse pixelmon's / soundmon's server aliases if you haven't made your own.
if [ ! -f "$REPO/servers.json" ]; then
    for src in "$HOME/pixelmon/servers.json" "$HOME/git/soundmon/servers.json"; do
        if [ -f "$src" ]; then
            cp "$src" "$REPO/servers.json"
            echo "   copied $src (same boxes, same port — the farm is shared)"
            break
        fi
    done
fi

# 4. Console font for the CP437 glyphs. We generate the block/shade characters
#    ourselves (Debian's console fonts are missing exactly those), but the
#    letters and box-drawing come from a PSF VGA font when one is available.
FONT_DESC="$("$COMFY/.venv/bin/python" -c \
    "import sys; sys.path.insert(0,'$REPO/custom_nodes/ansi_quantize'); \
     import cp437; print(cp437.font_source_description())" 2>/dev/null || true)"
if [ -n "$FONT_DESC" ]; then
    echo "==> CP437 text glyphs: $FONT_DESC"
    case "$FONT_DESC" in
        procedural*) echo "   For authentic VGA letters:  sudo apt install console-setup" ;;
    esac
else
    echo "==> couldn't probe the font (numpy missing from the venv?)"
fi

# 5. ansilove is optional — only needed if you want a second opinion on the
#    .ans files we emit. ansimon renders its own PNGs and does not shell out.
command -v ansilove >/dev/null 2>&1 \
    || echo "==> optional: sudo apt install ansilove   (independent .ans -> PNG check)"

# 6. render group — AMD/ROCm only (compute needs /dev/kfd, gated behind this group)
if [ "$GPU" = amd ] && ! id -nG | tr ' ' '\n' | grep -qx render; then
    echo "==> adding $USER to the 'render' group (ROCm GPU access)"
    sudo usermod -aG render "$USER"
    echo "   ⚠  LOG OUT AND BACK IN for this to take effect."
fi

cat <<EOF

✅ install done ($GPU).
   1) ./download-models.sh        # ~327 MB (ANSI Art Style XL LoRA)
   2) restart ComfyUI so it picks up the ansi_quantize node
   3) ansimon "a fierce dragon"

   Check everything with:  ansimon --doctor
   Note: make sure ~/.local/bin is on your PATH.
EOF
