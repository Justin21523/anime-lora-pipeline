#!/bin/bash
# Test Optimized Yokai Watch Pipeline
# Uses ALL frames from 3 episodes for character detection
# Only segments 2 episodes for background/character separation

set -e

echo "========================================="
echo "Yokai Watch Optimized Pipeline - Test"
echo "========================================="
echo ""
echo "Strategy:"
echo "  - Segment: 2 random episodes (save space)"
echo "  - Character Detection: Use ALL frames from 3 episodes"
echo "  - Train: 2 characters max"
echo ""

SCRIPT_DIR="/mnt/c/AI_LLM_projects/inazuma-eleven-lora/scripts/batch"
INPUT_FRAMES="/home/b0979/yokai_input_fast"
OUTPUT_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/optimized_test"

# Create test subdirectory with first 3 episodes
TEST_INPUT="/tmp/yokai_test_input"
rm -rf "$TEST_INPUT"
mkdir -p "$TEST_INPUT"

echo "📁 Creating test input with first 3 episodes..."
EPISODES=($(ls -d "$INPUT_FRAMES"/S1.* | head -3))
for ep in "${EPISODES[@]}"; do
    ep_name=$(basename "$ep")
    ln -s "$ep" "$TEST_INPUT/$ep_name"
    echo "  Linked: $ep_name"
done

echo ""
echo "🚀 Running optimized pipeline test..."
conda run -n blip2-env python3 "$SCRIPT_DIR/yokai_watch_optimized_pipeline.py" \
    --input-frames "$TEST_INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --segmentation-sample 2 \
    --min-cluster-size 25 \
    --max-characters 2 \
    --device cuda

echo ""
echo "========================================="
echo "✅ Test complete!"
echo "========================================="
echo ""
echo "Results in: $OUTPUT_DIR"
echo ""
