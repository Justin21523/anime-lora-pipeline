#!/bin/bash
# Quick test of Yokai Watch pipeline
# Process just 5 episodes to verify everything works

set -e  # Exit on error

echo "========================================="
echo "Yokai Watch Pipeline - Quick Test"
echo "========================================="
echo ""

# Paths
INPUT_FRAMES="/home/b0979/yokai_input_fast"
OUTPUT_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/quick_test"
SCRIPT_DIR="/mnt/c/AI_LLM_projects/inazuma-eleven-lora/scripts/batch"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run space-efficient pipeline with limited scope
echo "🚀 Starting pipeline test..."
echo "   - Processing first 5 episodes"
echo "   - Min cluster size: 25"
echo "   - Max characters to train: 2"
echo ""

conda run -n blip2-env python3 "$SCRIPT_DIR/yokai_watch_space_efficient.py" \
    --input-frames "$INPUT_FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    --episodes-per-batch 5 \
    --total-episodes 5 \
    --min-cluster-size 25 \
    --max-characters 2 \
    --device cuda

echo ""
echo "========================================="
echo "✅ Pipeline test complete!"
echo "========================================="
echo ""
echo "Check results in: $OUTPUT_DIR"
echo "Check log file: $OUTPUT_DIR/pipeline_log_*.txt"
echo ""
