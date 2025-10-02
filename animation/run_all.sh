#!/usr/bin/env bash
# Run all available PSD animations (2D and 3D) in one go
# Usage:
#   ./run_all.sh <input.psd> [--out-dir DIR] [--fps N]
#                 [--flicker-opts "..."] [--parallax-opts "..."] [--falling-opts "..."]
#                 [--3d-opts "..."] [--3d-flicker-opts "..."]
# Examples:
#   ./run_all.sh ../outputs/output_20250930_110724.psd --out-dir ./output --fps 24 \
#       --parallax-opts "--movement cardioid --duration 8" \
#       --falling-opts "--pattern top_to_bottom" \
#       --flicker-opts "--intensity 0.3" \
#       --3d-opts "--duration 10 --distance 1500" \
#       --3d-flicker-opts "--duration 15 --fps 12"

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
THREED_OPTS=""
THREED_FLICKER_OPTS=""

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.psd> [--out-dir DIR] [--fps N] [--flicker-opts \"...\"] [--parallax-opts \"...\"] [--falling-opts \"...\"] [--3d-opts \"...\"] [--3d-flicker-opts \"...\"]"
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
    --3d-opts)
      THREED_OPTS="$2"; shift 2 ;;
    --3d-flicker-opts)
      THREED_FLICKER_OPTS="$2"; shift 2 ;;
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
THREED_OUT="$OUT_DIR/${BASENAME_NOEXT}_3d.mp4"
THREED_FLICKER_OUT="$OUT_DIR/${BASENAME_NOEXT}_3d_flicker.mp4"

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
echo "\n[3/5] Running falling animation..."
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

# 4) 3D Camera (PyVista)
echo "\n[4/5] Running 3D camera animation (PyVista)..."
set +e
python "$SCRIPT_DIR/psd_3d_pyvista_mp4.py" "$PSD_FILE" -o "$THREED_OUT" --fps "$FPS" ${THREED_OPTS}
RET=$?
set -e
if [[ $RET -ne 0 ]]; then
  echo "  ❌ 3D camera failed (exit $RET)"
  FAIL=1
else
  echo "  ✅ 3D camera OK: $THREED_OUT"
fi

# 5) 3D Camera + Simple Appearance (PyVista)
echo "\n[5/5] Running 3D camera + simple appearance animation (PyVista)..."
set +e
python "$SCRIPT_DIR/psd_3d_flickering_mp4.py" "$PSD_FILE" -o "$THREED_FLICKER_OUT" --fps "$FPS" ${THREED_FLICKER_OPTS}
RET=$?
set -e
if [[ $RET -ne 0 ]]; then
  echo "  ❌ 3D flickering failed (exit $RET)"
  FAIL=1
else
  echo "  ✅ 3D flickering OK: $THREED_FLICKER_OUT"
fi

# Summary
echo "\n=== Summary ==="
[[ -f "$FLICKER_OUT" ]] && echo "Flicker (2D):        $FLICKER_OUT" || echo "Flicker (2D):        FAILED"
[[ -f "$PARALLAX_OUT" ]] && echo "Parallax (2D):       $PARALLAX_OUT" || echo "Parallax (2D):       FAILED"
[[ -f "$FALLING_OUT" ]] && echo "Falling (2D):        $FALLING_OUT" || echo "Falling (2D):        FAILED"
[[ -f "$THREED_OUT" ]] && echo "3D Camera:           $THREED_OUT" || echo "3D Camera:           FAILED"
[[ -f "$THREED_FLICKER_OUT" ]] && echo "3D + Appearance:     $THREED_FLICKER_OUT" || echo "3D + Appearance:     FAILED"

if [[ $FAIL -ne 0 ]]; then
  echo "\nSome tasks failed. See messages above."
  exit 2
fi

echo "\nAll tasks completed successfully."
exit 0
