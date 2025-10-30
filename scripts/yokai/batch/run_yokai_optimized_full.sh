#!/bin/bash
# Full Optimized Yokai Watch Pipeline
# Uses ALL 1M+ frames for character detection and LoRA training
# Only segments random 10 episodes for background/character separation

set -e

echo "========================================="
echo "Yokai Watch Optimized Full Pipeline"
echo "========================================="
echo ""

# Check tmux
if [ -z "$TMUX" ]; then
    echo "⚠️  Warning: Not running in tmux"
    echo "   Recommended: tmux new -s yokai_full"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

SCRIPT_DIR="/mnt/c/AI_LLM_projects/inazuma-eleven-lora/scripts/batch"
INPUT_FRAMES="/home/b0979/yokai_input_fast"
OUTPUT_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/optimized_full_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Input: ALL frames from $INPUT_FRAMES"
echo "  Output: $OUTPUT_DIR"
echo "  Segmentation: 10 random episodes (save space)"
echo "  Character Detection: ALL 1M+ frames"
echo "  LoRA Training: ALL detected character faces"
echo ""
echo "This will:"
echo "  1. Randomly sample 10 episodes for segmentation (~2-3 hours)"
echo "  2. Detect characters from ALL 1M+ frames (~5-8 hours)"
echo "  3. Train LoRA for each character (~1-2 hours each)"
echo ""
echo "Total estimated time: 2-3 days"
echo ""

read -p "Start full pipeline? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

START_TIME=$(date +%s)

echo "========================================="
echo "🚀 Pipeline started at $(date)"
echo "========================================="
echo ""

conda run -n blip2-env python3 "$SCRIPT_DIR/yokai_watch_optimized_pipeline.py" \
    --input-frames "$INPUT_FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    --segmentation-sample 10 \
    --min-cluster-size 50 \
    --device cuda \
    2>&1 | tee "$OUTPUT_DIR/full_pipeline.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================="
echo "✅ Pipeline completed at $(date)"
echo "   Total time: ${HOURS}h ${MINUTES}m"
echo "========================================="
echo ""
echo "Results:"
echo "  Output: $OUTPUT_DIR"
echo "  Log: $OUTPUT_DIR/full_pipeline.log"
echo "  Models: /mnt/c/AI_LLM_projects/ai_warehouse/models/lora/yokai-watch/"
echo ""
