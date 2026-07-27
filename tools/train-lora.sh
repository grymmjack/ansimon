#!/usr/bin/env bash
# Train a style LoRA on your own ANSI art, for use with `ansimon --lora`.
#
#   ./tools/train-lora.sh /path/to/ansi/src            # build dataset + train here
#   ./tools/train-lora.sh /path/to/ansi/src --prep     # build the dataset only
#   TOKEN=myname ./tools/train-lora.sh /path/to/src    # different trigger word
#
# This runs where you run it. To train on another box (recommended — pick the
# newest CUDA card you have, see the note below), run it there, or run --prep
# here and rsync the dataset over.
#
# GPU choice matters more than VRAM here. Ampere and later have real bf16
# tensor cores; Pascal (GTX 10xx / Titan Xp) does not, and trains SDXL far
# slower even with more VRAM. An 8 GB RTX 3070 beats a 12 GB Titan Xp for this.
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
SRC="${1:-}"
TOKEN="${TOKEN:-grymmjack}"
WORK="${WORK:-$HOME/ansimon-lora}"
SCRIPTS="${SD_SCRIPTS:-$HOME/sd-scripts}"
BASE="${BASE:-$HOME/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors}"

[ -n "$SRC" ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ -d "$SRC" ] || { echo "no such folder: $SRC" >&2; exit 1; }

PREP_ONLY=0
[ "${2:-}" = "--prep" ] && PREP_ONLY=1

PY="$HOME/ComfyUI/.venv/bin/python"
[ -x "$PY" ] || PY=python3

echo "==> dataset:  $WORK/dataset"
"$PY" "$REPO/tools/build-lora-dataset.py" "$SRC" "$WORK/dataset" --token "$TOKEN"

if [ "$PREP_ONLY" = 1 ]; then
    cat <<EOF

✅ dataset only. To train on another machine:
   rsync -a "$WORK/dataset/" otherbox:~/ansimon-lora/dataset/
   then run this script there with --prep omitted.
EOF
    exit 0
fi

# --- sd-scripts ----------------------------------------------------------
if [ ! -d "$SCRIPTS/.git" ]; then
    echo "==> cloning kohya sd-scripts -> $SCRIPTS"
    git clone https://github.com/kohya-ss/sd-scripts.git "$SCRIPTS"
fi
if [ ! -x "$SCRIPTS/.venv/bin/python" ]; then
    echo "==> building sd-scripts venv (torch + bitsandbytes, a few GB)"
    python3 -m venv "$SCRIPTS/.venv"
    "$SCRIPTS/.venv/bin/python" -m pip install -q --upgrade pip
    # cu124 wheels cover Ampere and later.
    "$SCRIPTS/.venv/bin/python" -m pip install -q \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124
    "$SCRIPTS/.venv/bin/python" -m pip install -q -r "$SCRIPTS/requirements.txt"
    "$SCRIPTS/.venv/bin/python" -m pip install -q bitsandbytes
fi
TPY="$SCRIPTS/.venv/bin/python"

# --- sanity: is the GPU actually free? -----------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "==> GPU: $NAME — ${USED}/${TOTAL} MiB in use"
    if [ "$USED" -gt 1500 ]; then
        echo "    ⚠  Something is already holding VRAM (ComfyUI?). SDXL LoRA on 8 GB"
        echo "       needs essentially all of it. Stop that first:"
        echo "           PID=\$(ss -ltnp | grep 8188 | grep -oP 'pid=\\K[0-9]+' | head -1); kill \$PID"
    fi
fi

[ -f "$BASE" ] || { echo "SDXL base not found at $BASE (set BASE=...)" >&2; exit 1; }

CFG="$WORK/grymmjack-sdxl.toml"
mkdir -p "$WORK/output" "$WORK/logs"
sed -e "s|^pretrained_model_name_or_path = .*|pretrained_model_name_or_path = \"$BASE\"|" \
    -e "s|^output_name = .*|output_name = \"$TOKEN\"|" \
    -e "s|^output_dir = .*|output_dir = \"$WORK/output\"|" \
    -e "s|^logging_dir = .*|logging_dir = \"$WORK/logs\"|" \
    "$REPO/tools/lora/grymmjack-sdxl.toml" > "$CFG"

echo "==> training ($TOKEN) — config $CFG"
cd "$SCRIPTS"
"$SCRIPTS/.venv/bin/accelerate" launch --num_cpu_threads_per_process 4 \
    sdxl_train_network.py --config_file "$CFG" \
    --train_data_dir "$WORK/dataset"

cat <<EOF

✅ done. LoRA(s) in $WORK/output

   Install and use it:
       cp $WORK/output/$TOKEN.safetensors ~/ComfyUI/models/loras/
       ansimon "a dragon" --lora $TOKEN.safetensors --lora-strength 0.9

   The trigger word is baked into every caption, so put it in the prompt:
       ansimon "$TOKEN, a skull logo" --lora $TOKEN.safetensors

   Compare against the generic LoRA at the same seed to judge it:
       ansimon "a skull" --seed 900                      # stock ansi-art-xl
       ansimon "a skull" --seed 900 --lora $TOKEN.safetensors
EOF
