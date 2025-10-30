#!/bin/bash
# Complete Audio Processing Pipeline for Inazuma Eleven
#
# This script runs the complete audio processing pipeline:
# 1. Extract audio from video files
# 2. Separate vocals from background music/SFX
# 3. Analyze voice characteristics
#
# Usage: bash scripts/batch/inazuma_eleven_audio.sh

# Configuration
SERIES="inazuma-eleven"
BASE_DIR="/mnt/c/AI_LLM_projects/ai_warehouse"
VIDEO_DIR="$BASE_DIR/training_data/$SERIES/raw_videos"
CACHE_DIR="$BASE_DIR/cache/$SERIES"

# Audio extraction settings
AUDIO_FORMAT="wav"
SAMPLE_RATE=44100
CHANNELS=1
WORKERS=16

# Voice separation settings
DEMUCS_MODEL="htdemucs"
SHIFTS=5  # Higher quality for training data
DEVICE="cuda"

echo "=========================================="
echo "Inazuma Eleven Audio Processing Pipeline"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Series: $SERIES"
echo "  Video directory: $VIDEO_DIR"
echo "  Cache directory: $CACHE_DIR"
echo "  Workers: $WORKERS"
echo "  Device: $DEVICE"
echo ""
echo "Pipeline steps:"
echo "  [1/3] Audio extraction"
echo "  [2/3] Vocal separation"
echo "  [3/3] Voice analysis"
echo ""
echo "Estimated time: 4-6 hours total"
echo "=========================================="
echo ""
read -p "Press Enter to start..."
echo ""

# Step 1: Extract audio from videos
echo "=========================================="
echo "[1/3] AUDIO EXTRACTION"
echo "=========================================="
echo ""
echo "Extracting audio from all episode videos..."
echo "Format: $AUDIO_FORMAT @ ${SAMPLE_RATE}Hz (mono)"
echo ""

conda run -n blip2-env python3 scripts/tools/audio_extractor.py \
  "$VIDEO_DIR" \
  --output-dir "$CACHE_DIR/audio" \
  --format "$AUDIO_FORMAT" \
  --sample-rate "$SAMPLE_RATE" \
  --channels "$CHANNELS" \
  --workers "$WORKERS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Audio extraction complete!"
    echo ""

    # Show statistics
    echo "Statistics:"
    AUDIO_COUNT=$(find "$CACHE_DIR/audio" -name "*.wav" | wc -l)
    AUDIO_SIZE=$(du -sh "$CACHE_DIR/audio" | cut -f1)
    echo "  Files extracted: $AUDIO_COUNT"
    echo "  Total size: $AUDIO_SIZE"
    echo ""
else
    echo ""
    echo "✗ Audio extraction failed!"
    exit 1
fi

# Step 2: Separate vocals from background
echo "=========================================="
echo "[2/3] VOCAL SEPARATION"
echo "=========================================="
echo ""
echo "Separating vocals from background music and sound effects..."
echo "Model: $DEMUCS_MODEL (quality: $SHIFTS shifts)"
echo "This may take 2-3 hours with GPU..."
echo ""

conda run -n blip2-env python3 scripts/tools/voice_separator.py \
  "$CACHE_DIR/audio" \
  --output-dir "$CACHE_DIR/separated" \
  --model "$DEMUCS_MODEL" \
  --shifts "$SHIFTS" \
  --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Vocal separation complete!"
    echo ""

    # Show statistics
    echo "Statistics:"
    VOCALS_COUNT=$(find "$CACHE_DIR/separated/vocals" -name "*.wav" 2>/dev/null | wc -l)
    VOCALS_SIZE=$(du -sh "$CACHE_DIR/separated/vocals" 2>/dev/null | cut -f1)
    echo "  Vocals extracted: $VOCALS_COUNT"
    echo "  Total size: $VOCALS_SIZE"
    echo ""
else
    echo ""
    echo "✗ Vocal separation failed!"
    exit 1
fi

# Step 3: Analyze voice characteristics
echo "=========================================="
echo "[3/3] VOICE ANALYSIS"
echo "=========================================="
echo ""
echo "Analyzing voice characteristics (pitch, formants, etc.)..."
echo "This will create visualizations and CSV results..."
echo ""

conda run -n blip2-env python3 scripts/tools/voice_analyzer.py \
  "$CACHE_DIR/separated/vocals" \
  --output-csv "$CACHE_DIR/voice_analysis.csv" \
  --formants \
  --visualize \
  --viz-dir "$CACHE_DIR/visualizations"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Voice analysis complete!"
    echo ""

    # Show statistics
    echo "Statistics:"
    VIZ_COUNT=$(find "$CACHE_DIR/visualizations" -name "*.png" 2>/dev/null | wc -l)
    echo "  Visualizations created: $VIZ_COUNT"
    echo "  CSV results: voice_analysis.csv"
    echo ""
else
    echo ""
    echo "✗ Voice analysis failed!"
    echo "Note: This step may fail if praat-parselmouth is not installed"
    echo "Try: pip install praat-parselmouth"
    exit 1
fi

# Final summary
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "All processing steps completed successfully!"
echo ""
echo "Output locations:"
echo "  📁 Audio files:     $CACHE_DIR/audio"
echo "  🎤 Vocals only:     $CACHE_DIR/separated/vocals"
echo "  📊 Analysis CSV:    $CACHE_DIR/voice_analysis.csv"
echo "  📈 Visualizations:  $CACHE_DIR/visualizations"
echo ""
echo "Next steps:"
echo "  1. Review voice_analysis.csv to see voice characteristics"
echo "  2. Check visualizations to identify different characters"
echo "  3. Group similar voices by character name"
echo "  4. Select high-quality samples for voice cloning"
echo ""
echo "To view a sample visualization:"
echo "  ls $CACHE_DIR/visualizations/*.png | head -1 | xargs -I {} echo {}"
echo ""
echo "To view analysis results:"
echo "  head -20 $CACHE_DIR/voice_analysis.csv"
echo ""
