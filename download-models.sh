#!/usr/bin/env bash
# Download the models ansimon needs into ~/ComfyUI/models/.
#
# ansimon needs exactly one thing pixelmon doesn't already have: an ANSI-art
# LoRA. Everything else (the SDXL base, the LCM LoRA used by --fast) is shared
# with pixelmon, so on a box that already runs pixelmon this fetches ~327 MB
# and nothing else.
#
#   ./download-models.sh              # SDXL LoRA (recommended)
#   ./download-models.sh --sd15       # also grab the SD1.5 version
#   ./download-models.sh --base       # also grab the SDXL base checkpoint
set -euo pipefail

COMFY="${COMFYUI_DIR:-$HOME/ComfyUI}"
LORA="$COMFY/models/loras"
CKPT="$COMFY/models/checkpoints"
mkdir -p "$LORA" "$CKPT"

WANT_SD15=0; WANT_BASE=0
for arg in "$@"; do
    case "$arg" in
        --sd15) WANT_SD15=1 ;;
        --base) WANT_BASE=1 ;;
        --all)  WANT_SD15=1; WANT_BASE=1 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

# Civitai gates most downloads behind an API token. Set CIVITAI_TOKEN to make
# this unattended; without one you get an HTML login page instead of a model,
# which we detect below rather than handing ComfyUI a 4 KB "safetensors" file.
AUTH=()
[ -n "${CIVITAI_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $CIVITAI_TOKEN")

get() {  # url  dest  human-name
    local url="$1" dest="$2" name="${3:-$(basename "$2")}"
    if [ -f "$dest" ]; then echo "✓ already have $name"; return; fi
    echo "↓ downloading $name ..."
    # Download to .part and rename only on success, so an interrupted transfer
    # can never leave a truncated file at the real path — where the check above
    # would treat it as complete next run and hand ComfyUI a corrupt model.
    curl -L --fail -C - ${AUTH[@]+"${AUTH[@]}"} -o "$dest.part" "$url"
    # A Civitai login redirect is small and starts with '<'; a safetensors file
    # is large and starts with a JSON header length. Catch the former early.
    if [ "$(stat -c%s "$dest.part" 2>/dev/null || stat -f%z "$dest.part")" -lt 1000000 ]; then
        rm -f "$dest.part"
        cat >&2 <<'EOF'

❌ That download came back too small — almost certainly a Civitai login page.

   Civitai requires an API token for model downloads. Make one at
       https://civitai.com/user/account   (API Keys -> Add API key)
   then re-run:
       CIVITAI_TOKEN=xxxxxxxx ./download-models.sh

   Or download by hand and drop it in place:
       https://civitai.com/models/185538/ansi-art-xl
       -> save as  ~/ComfyUI/models/loras/ansi-art-xl.safetensors
EOF
        exit 1
    fi
    mv "$dest.part" "$dest"
}

# ANSI Art Style XL — the whole ballgame, same role Pixel Art XL plays in
# pixelmon. Trigger word is `ansiart` (ansimon puts it in the prompt for you).
# The author's note: the XL version has better character/block accuracy and
# proper 4-bit palettes than the 1.5, at the cost of being "a bit dull".
get "https://civitai.com/api/download/models/208294" \
    "$LORA/ansi-art-xl.safetensors" "ANSI Art Style XL LoRA (327 MB)"

if [ "$WANT_SD15" = 1 ]; then
    # The SD1.5 original. Punchier and more chaotic than the XL; some people
    # prefer it. Use with  --base <an SD1.5 checkpoint> --lora ansi-art-15.safetensors
    get "https://civitai.com/api/download/models/207417" \
        "$LORA/ansi-art-15.safetensors" "ANSI Art Style LoRA, SD1.5 (54 MB)"
fi

if [ "$WANT_BASE" = 1 ]; then
    # SDXL base — shared with pixelmon, so skip it if you already have it.
    get "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
        "$CKPT/sd_xl_base_1.0.safetensors" "SDXL base 1.0 (6.9 GB)"
fi

cat <<EOF

✅ models ready in $LORA
   Restart ComfyUI, then:  ansimon "a fierce dragon"

   Missing the SDXL base or the LCM LoRA? They're shared with pixelmon:
       ~/pixelmon/download-models.sh
EOF
