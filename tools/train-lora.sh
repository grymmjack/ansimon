#!/usr/bin/env bash
# Train a style LoRA on your own ANSI art, for use with `ansimon --lora`.
#
#   ./tools/train-lora.sh /path/to/ansi/src            # SDXL LoRA, build + train
#   ./tools/train-lora.sh /path/to/ansi/src --sd15     # SD 1.5 LoRA (fast run)
#   ./tools/train-lora.sh /path/to/ansi/src --prep     # build the dataset only
#   TOKEN=myname ./tools/train-lora.sh /path/to/src    # different trigger word
#
# Which one to run
#   --sd15  is the CHEAP ITERATION run. SD1.5's UNet is 860M params against
#           SDXL's 2.6B, so on a 12 GB card it needs no memory tricks, takes a
#           batch of 4, and finishes in a fraction of the time. Use it to find
#           the learning rate and epoch count, and to answer "is there enough
#           signal in this corpus at all?" before committing a long SDXL job.
#   default is the QUALITY run: SDXL, 8 GB-tuned, slow.
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

[ -n "$SRC" ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ -d "$SRC" ] || { echo "no such folder: $SRC" >&2; exit 1; }

PREP_ONLY=0; SD15=0
for arg in "${@:2}"; do
    case "$arg" in
        --prep) PREP_ONLY=1 ;;
        --sd15) SD15=1 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

# SD 1.5 trains at the art's NATIVE size: an 80x24 screen is exactly 640x384,
# both /64-divisible. SDXL wants it doubled to 1280x768, which is one of its
# own bucket sizes. Either way nothing is cropped or resampled.
if [ "$SD15" = 1 ]; then
    UPSCALE=1; CFG_SRC="grymmjack-sd15.toml"; TRAIN_PY="train_network.py"
    BASE="${BASE:-$HOME/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.safetensors}"
    SUFFIX="-15"
else
    UPSCALE=2; CFG_SRC="grymmjack-sdxl.toml"; TRAIN_PY="sdxl_train_network.py"
    BASE="${BASE:-$HOME/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors}"
    SUFFIX=""
fi

PY="$HOME/ComfyUI/.venv/bin/python"
[ -x "$PY" ] || PY=python3

echo "==> dataset:  $WORK/dataset"
"$PY" "$REPO/tools/build-lora-dataset.py" "$SRC" "$WORK/dataset" \
    --token "$TOKEN" --upscale "$UPSCALE"

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
    NEED=$([ "$SD15" = 1 ] && echo 6000 || echo 7000)
    FREE=$(( TOTAL - USED ))
    if [ "$FREE" -lt "$NEED" ]; then
        echo "    ⚠  Only ${FREE} MiB free; this run wants ~${NEED} MiB."
        echo "       Something is holding VRAM (ComfyUI?). Stop it first:"
        echo "           PID=\$(ss -ltnp | grep 8188 | grep -oP 'pid=\\K[0-9]+' | head -1); kill \$PID"
    fi
    case "$NAME" in
        *"TITAN Xp"*|*"1080"*|*"1070"*|*"1060"*)
            if [ "$SD15" != 1 ]; then
                echo "    ⚠  $NAME is Pascal: no bf16, fp16 at 1/64 rate. It will train"
                echo "       SDXL very slowly. Consider --sd15 on this box, and save SDXL"
                echo "       for an Ampere or newer card."
            fi ;;
    esac
fi

[ -f "$BASE" ] || { echo "base checkpoint not found at $BASE (set BASE=...)" >&2; exit 1; }

CFG="$WORK/$CFG_SRC"
mkdir -p "$WORK/output" "$WORK/logs"
sed -e "s|^pretrained_model_name_or_path = .*|pretrained_model_name_or_path = \"$BASE\"|" \
    -e "s|^output_name = .*|output_name = \"$TOKEN$SUFFIX\"|" \
    -e "s|^output_dir = .*|output_dir = \"$WORK/output\"|" \
    -e "s|^logging_dir = .*|logging_dir = \"$WORK/logs\"|" \
    "$REPO/tools/lora/$CFG_SRC" > "$CFG"

echo "==> training ($TOKEN$SUFFIX, $([ "$SD15" = 1 ] && echo 'SD 1.5' || echo SDXL)) — config $CFG"
cd "$SCRIPTS"
"$SCRIPTS/.venv/bin/accelerate" launch --num_cpu_threads_per_process 4 \
    "$TRAIN_PY" --config_file "$CFG" \
    --train_data_dir "$WORK/dataset"

cat <<EOF

✅ done. LoRA(s) in $WORK/output

   Install and use it:
       cp $WORK/output/$TOKEN$SUFFIX.safetensors ~/ComfyUI/models/loras/
       ansimon "a dragon" --lora $TOKEN$SUFFIX.safetensors --lora-strength 0.9

   The trigger word is baked into every caption, so put it in the prompt:
       ansimon "$TOKEN, a skull logo" --lora $TOKEN$SUFFIX.safetensors$([ "$SD15" = 1 ] && echo " \\
           --base v1-5-pruned-emaonly.safetensors" || true)

   Compare against the generic LoRA at the same seed to judge it:
       ansimon "a skull" --seed 900                      # stock ansi-art-xl
       ansimon "a skull" --seed 900 --lora $TOKEN$SUFFIX.safetensors
EOF
