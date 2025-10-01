#!/usr/bin/env bash
# Run flickering, parallax, and falling animations in one go
# Usage:
#   ./run_all.sh <input.psd> [--out-dir DIR] [--fps N]
#                 [--flicker-opts "..."] [--parallax-opts "..."] [--falling-opts "..."]
# Examples:
#   ./run_all.sh ../outputs/output_20250930_110724.psd --out-dir ./output --fps 24 \
#       --parallax-opts "--movement cardioid --duration 8" \
#       --falling-opts "--pattern top_to_bottom" \
#       --flicker-opts "--intensity 0.3"

set -euo pipefail

# Resolve script directory (so we can call sibling Python scripts reliably)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
OUT_DIR="${SCRIPT_DIR}/output"
FPS=24
FLICKER_OPTS=""
PARALLAX_OPTS=""
FALLING_OPTS=""

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.psd> [--out-dir DIR] [--fps N] [--flicker-opts \"...\"] [--parallax-opts \"...\"] [--falling-opts \"...\"]"
  exit 1
fi

PSD_FILE="$1"; shift || true

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --fps)
      FPS="$2"; shift 2 ;;
    --flicker-opts)
      FLICKER_OPTS="$2"; shift 2 ;;
    --parallax-opts)
      PARALLAX_OPTS="$2"; shift 2 ;;
    --falling-opts)
      FALLING_OPTS="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Validate PSD
if [[ ! -f "$PSD_FILE" ]]; then
  echo "Error: PSD file not found: $PSD_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Derive base name (without extension)
BASENAME="$(basename "$PSD_FILE")"
BASENAME_NOEXT="${BASENAME%.*}"

# Outputs
FLICKER_OUT="$OUT_DIR/${BASENAME_NOEXT}_flicker.mp4"
PARALLAX_OUT="$OUT_DIR/${BASENAME_NOEXT}_parallax.mp4"
FALLING_OUT="$OUT_DIR/${BASENAME_NOEXT}_falling.mp4"

# Status bookkeeping
FAIL=0

# 1) Flickering
echo "\n[1/3] Running flickering animation..."
set +e
python "$SCRIPT_DIR/psd_flickering_mp4.py" "$PSD_FILE" -o "$FLICKER_OUT" --fps "$FPS" ${FLICKER_OPTS} 
RET=$?
set -e
if [[ $RET -ne 0 ]]; then
  echo "  ❌ Flickering failed (exit $RET)"
  FAIL=1
else
  echo "  ✅ Flickering OK: $FLICKER_OUT"
fi

# 2) Parallax
echo "\n[2/3] Running parallax animation..."
set +e
python "$SCRIPT_DIR/psd_parallax_mp4.py" "$PSD_FILE" -o "$PARALLAX_OUT" --fps "$FPS" ${PARALLAX_OPTS}
RET=$?
set -e
if [[ $RET -ne 0 ]]; then
  echo "  ❌ Parallax failed (exit $RET)"
  FAIL=1
else
  echo "  ✅ Parallax OK: $PARALLAX_OUT"
fi

# 3) Falling
echo "\n[3/3] Running falling animation..."
set +e
python "$SCRIPT_DIR/psd_falling_mp4.py" "$PSD_FILE" -o "$FALLING_OUT" --fps "$FPS" ${FALLING_OPTS}
RET=$?
set -e
if [[ $RET -ne 0 ]]; then
  echo "  ❌ Falling failed (exit $RET)"
  FAIL=1
else
  echo "  ✅ Falling OK: $FALLING_OUT"
fi

# Summary
echo "\n=== Summary ==="
[[ -f "$FLICKER_OUT" ]] && echo "Flicker:  $FLICKER_OUT" || echo "Flicker:  FAILED"
[[ -f "$PARALLAX_OUT" ]] && echo "Parallax: $PARALLAX_OUT" || echo "Parallax: FAILED"
[[ -f "$FALLING_OUT" ]] && echo "Falling:  $FALLING_OUT" || echo "Falling:  FAILED"

if [[ $FAIL -ne 0 ]]; then
  echo "\nSome tasks failed. See messages above."
  exit 2
fi

echo "\nAll tasks completed successfully."
exit 0
