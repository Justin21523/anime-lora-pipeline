#!/bin/bash
# Test Ultra-Fast Parallel Pipeline
# Uses 16 CPU workers + GPU for maximum performance

set -e

echo "========================================="
echo "🚀 Ultra-Fast Parallel Pipeline - TEST"
echo "========================================="
echo ""
echo "Performance:"
echo "  - 16 CPU workers for segmentation"
echo "  - GPU batch processing for clustering"
echo "  - 8 DataLoader workers"
echo "  - Expected: 10-15x faster"
echo ""

SCRIPT_DIR="/mnt/c/AI_LLM_projects/inazuma-eleven-lora/scripts/batch"
INPUT_FRAMES="/home/b0979/yokai_input_fast"
OUTPUT_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/parallel_test"

# Create test input (first 3 episodes)
TEST_INPUT="/tmp/yokai_parallel_test_input"
rm -rf "$TEST_INPUT"
mkdir -p "$TEST_INPUT"

echo "📁 Creating test input (3 episodes)..."
EPISODES=($(ls -d "$INPUT_FRAMES"/S1.* | head -3))
for ep in "${EPISODES[@]}"; do
    ep_name=$(basename "$ep")
    ln -s "$ep" "$TEST_INPUT/$ep_name"
    echo "  Linked: $ep_name"
done

echo ""
echo "🚀 Launching ultra-fast parallel pipeline..."
conda run -n blip2-env python3 "$SCRIPT_DIR/yokai_parallel_pipeline.py" \
    --input-frames "$TEST_INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --segmentation-sample 2 \
    --min-cluster-size 25 \
    --max-characters 2 \
    --num-seg-workers 16 \
    --batch-size 64 \
    --num-dataloader-workers 8 \
    --device cuda

echo ""
echo "========================================="
echo "✅ Test complete!"
echo "========================================="
echo ""
echo "Results: $OUTPUT_DIR"
echo ""
