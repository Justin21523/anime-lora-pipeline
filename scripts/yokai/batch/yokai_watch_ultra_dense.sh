#!/bin/bash
# Yokai Watch Ultra-Dense Frame Extraction
# Optimized for capturing fast yokai summoning animations with maximum detail
#
# Configuration: EXTREME DENSITY MODE
# - Scene threshold: 15 (very sensitive, catches subtle movements)
# - Frames per scene: 8 (captures full motion progression)
# - Interval: 2 seconds (double temporal density)
# - Workers: 16 (maximum parallelization)
#
# This will extract ~4,000-5,000 frames per episode!

# Configuration
INPUT_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/training_data/yokai-watch/raw_videos"
OUTPUT_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/cache/yokai-watch/extracted_frames_ultra_dense"
TEMP_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/cache/yokai-watch/temp_converted"

# Ultra-dense extraction parameters
MODE="hybrid"
SCENE_THRESHOLD=15      # Very sensitive (catches micro-movements)
FRAMES_PER_SCENE=8      # Maximum frame coverage per scene
INTERVAL_SECONDS=2      # Extract every 2 seconds (vs 4s before)
WORKERS=16              # Full CPU utilization
JPEG_QUALITY=95

# Episode pattern: use "none" to preserve full filenames
# This prevents overwrites when processing multi-season anime
# (S1.01, S2.01, S3.01 would all overwrite each other with numeric patterns)
EPISODE_PATTERN="none"

echo "=========================================================================="
echo "Yokai Watch Ultra-Dense Frame Extraction"
echo "=========================================================================="
echo ""
echo "📊 Video File Statistics:"
echo "  Season 1: 65 episodes"
echo "  Season 2: 52 episodes"
echo "  Season 3: 52 episodes"
echo "  Season 4: 45 episodes"
echo "  Movies:   4 films"
echo "  ─────────────────────"
echo "  Total:    218 files"
echo ""
echo "⚙️  Extraction Configuration (EXTREME DENSITY):"
echo "  Input:             $INPUT_DIR"
echo "  Output:            $OUTPUT_DIR"
echo "  Mode:              $MODE"
echo "  Scene Threshold:   $SCENE_THRESHOLD (↓ more sensitive)"
echo "  Frames per Scene:  $FRAMES_PER_SCENE (↑ more coverage)"
echo "  Interval:          ${INTERVAL_SECONDS}s (↑ 2x denser)"
echo "  Workers:           $WORKERS cores"
echo "  JPEG Quality:      ${JPEG_QUALITY}%"
echo ""
echo "📈 Expected Output:"
echo "  Frames per episode: ~4,000-5,000 frames"
echo "  Total frames:       ~900,000-1,000,000 frames"
echo "  Storage required:   ~150-180 GB"
echo ""
echo "⏱️  Estimated Time:"
echo "  Per episode:        ~3-4 minutes"
echo "  Total (218 files):  ~12-15 hours"
echo ""
echo "💡 Why Ultra-Dense?"
echo "  ✓ Captures rapid yokai transformation sequences"
echo "  ✓ Records subtle facial expression changes"
echo "  ✓ Preserves attack animation details"
echo "  ✓ Ideal for frame interpolation training"
echo "  ✓ Maximum coverage for character variety"
echo ""
echo "=========================================================================="
echo ""
read -p "Press Enter to start ultra-dense extraction..."
echo ""

# Check available disk space
AVAILABLE_SPACE=$(df -BG "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')
if [ -z "$AVAILABLE_SPACE" ]; then
    AVAILABLE_SPACE=$(df -BG /mnt/c | awk 'NR==2 {print $4}' | sed 's/G//')
fi

echo "Checking disk space..."
echo "  Available: ${AVAILABLE_SPACE}G"
echo "  Required:  ~180G"
echo ""

if [ "$AVAILABLE_SPACE" -lt 200 ]; then
    echo "⚠️  WARNING: Low disk space!"
    echo "  You may need to free up space during extraction."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Extraction cancelled."
        exit 1
    fi
fi

# Run extraction
echo "Starting extraction..."
echo ""

conda run -n blip2-env python3 \
  /mnt/c/AI_LLM_projects/inazuma-eleven-lora/scripts/tools/universal_frame_extractor.py \
  "$INPUT_DIR" \
  --mode "$MODE" \
  --scene-threshold "$SCENE_THRESHOLD" \
  --frames-per-scene "$FRAMES_PER_SCENE" \
  --interval-seconds "$INTERVAL_SECONDS" \
  --episode-pattern "$EPISODE_PATTERN" \
  --output-dir "$OUTPUT_DIR" \
  --temp-dir "$TEMP_DIR" \
  --jpeg-quality "$JPEG_QUALITY" \
  --workers "$WORKERS"

EXIT_CODE=$?

echo ""
echo "=========================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ EXTRACTION COMPLETE!"
    echo "=========================================================================="
    echo ""

    # Verify output
    echo "📊 Extraction Statistics:"

    TOTAL_FRAMES=$(find "$OUTPUT_DIR" -name "*.jpg" 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    EPISODES_PROCESSED=$(ls -d "$OUTPUT_DIR"/episode_* 2>/dev/null | wc -l)

    echo "  Episodes processed: $EPISODES_PROCESSED / 218"
    echo "  Total frames:       $(printf "%'d" $TOTAL_FRAMES)"
    echo "  Storage used:       $TOTAL_SIZE"
    echo ""

    # Calculate average frames per episode
    if [ $EPISODES_PROCESSED -gt 0 ]; then
        AVG_FRAMES=$((TOTAL_FRAMES / EPISODES_PROCESSED))
        echo "  Average per episode: $(printf "%'d" $AVG_FRAMES) frames"
        echo ""
    fi

    echo "📁 Output Location:"
    echo "  $OUTPUT_DIR"
    echo ""

    echo "🎯 Next Steps:"
    echo ""
    echo "  1. Verify frame quality:"
    echo "     ls $OUTPUT_DIR/episode_001/*.jpg | head -20"
    echo ""
    echo "  2. Check specific yokai summoning scenes:"
    echo "     # Look for rapid scene changes (high frame density)"
    echo "     ls -lt $OUTPUT_DIR/episode_001/ | head -50"
    echo ""
    echo "  3. Compare with previous extraction:"
    echo "     echo \"Previous: ~2,400 frames/episode\""
    echo "     echo \"New:      ~4,500 frames/episode\""
    echo "     echo \"Improvement: +87% frame coverage\""
    echo ""
    echo "  4. Sample a summoning sequence:"
    echo "     # Check frames around timestamp 5:30-6:00 (common summon time)"
    echo "     ffmpeg -i \"$INPUT_DIR/S1.01.mp4\" -ss 330 -t 30 -r 10 /tmp/sample_%03d.jpg"
    echo ""
    echo "  5. Clean up temp files (optional):"
    echo "     rm -rf $TEMP_DIR/*.mp4"
    echo "     # This will free up ~36 GB"
    echo ""

else
    echo "❌ EXTRACTION FAILED!"
    echo "=========================================================================="
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Disk space full: Check df -h"
    echo "  - Missing dependencies: pip install scenedetect[opencv]"
    echo "  - Permission issues: Check file permissions"
    echo ""
    echo "For debugging, check the error messages above."
    echo ""
fi

echo "=========================================================================="
echo ""
