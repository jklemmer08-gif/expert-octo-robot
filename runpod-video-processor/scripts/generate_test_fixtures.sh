#!/usr/bin/env bash
# Generate small test videos for unit/integration tests.
# Requires: ffmpeg
# Output: tests/fixtures/

set -euo pipefail

FIXTURES_DIR="$(cd "$(dirname "$0")/../tests/fixtures" && pwd)"
mkdir -p "$FIXTURES_DIR"

echo "Generating test fixtures in $FIXTURES_DIR ..."

# --- 2D test: 10 seconds, 720p, H.264, with audio tone ---
ffmpeg -y -f lavfi -i "testsrc=duration=10:size=1280x720:rate=30" \
       -f lavfi -i "sine=frequency=440:duration=10" \
       -c:v libx264 -preset ultrafast -crf 23 \
       -c:a aac -b:a 128k \
       -pix_fmt yuv420p \
       "$FIXTURES_DIR/test_2d_720p.mp4" 2>/dev/null
echo "  [OK] test_2d_720p.mp4"

# --- VR SBS test: 10 seconds, 1280x640 (SBS 180°, 2:1 aspect) ---
ffmpeg -y -f lavfi -i "testsrc=duration=10:size=1280x640:rate=30" \
       -f lavfi -i "sine=frequency=440:duration=10" \
       -c:v libx264 -preset ultrafast -crf 23 \
       -c:a aac -b:a 128k \
       -pix_fmt yuv420p \
       -metadata:s:v stereo_mode=side_by_side \
       "$FIXTURES_DIR/test_SBS_vr.mp4" 2>/dev/null
echo "  [OK] test_SBS_vr.mp4"

# --- VR OU test: 10 seconds, 640x640 (OU 180°, 1:2 per-eye) ---
ffmpeg -y -f lavfi -i "testsrc=duration=10:size=640x1280:rate=30" \
       -f lavfi -i "sine=frequency=440:duration=10" \
       -c:v libx264 -preset ultrafast -crf 23 \
       -c:a aac -b:a 128k \
       -pix_fmt yuv420p \
       -metadata:s:v stereo_mode=top_bottom \
       "$FIXTURES_DIR/test_OU_vr.mp4" 2>/dev/null
echo "  [OK] test_OU_vr.mp4"

# --- Corrupt test: truncated file ---
ffmpeg -y -f lavfi -i "testsrc=duration=5:size=320x240:rate=30" \
       -c:v libx264 -preset ultrafast -crf 23 \
       -pix_fmt yuv420p \
       "$FIXTURES_DIR/test_full_temp.mp4" 2>/dev/null
# Truncate to 50% of file size to simulate corruption
FILE_SIZE=$(stat -c%s "$FIXTURES_DIR/test_full_temp.mp4" 2>/dev/null || stat -f%z "$FIXTURES_DIR/test_full_temp.mp4")
HALF=$((FILE_SIZE / 2))
dd if="$FIXTURES_DIR/test_full_temp.mp4" of="$FIXTURES_DIR/test_corrupt.mp4" bs=1 count="$HALF" 2>/dev/null
rm -f "$FIXTURES_DIR/test_full_temp.mp4"
echo "  [OK] test_corrupt.mp4"

# --- Small 2D test: 3 seconds, 64x64, for fast upscaler testing ---
ffmpeg -y -f lavfi -i "testsrc=duration=3:size=64x64:rate=10" \
       -c:v libx264 -preset ultrafast -crf 23 \
       -pix_fmt yuv420p \
       "$FIXTURES_DIR/test_tiny.mp4" 2>/dev/null
echo "  [OK] test_tiny.mp4"

echo ""
echo "All fixtures generated in $FIXTURES_DIR"
ls -lh "$FIXTURES_DIR"
