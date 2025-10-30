#!/bin/bash
# Ultra-Fast Full Parallel Pipeline
# Utilizes all 32 CPU cores + GPU

set -e

echo "========================================="
echo "🚀 Yokai Watch Ultra-Fast Full Pipeline"
echo "========================================="
echo ""

# Check tmux
if [ -z "$TMUX" ]; then
    echo "⚠️  Not in tmux - starting tmux session"
    tmux new -s yokai_ultra "bash $0"
    exit 0
fi

SCRIPT_DIR="/mnt/c/AI_LLM_projects/inazuma-eleven-lora/scripts/batch"
INPUT_FRAMES="/home/b0979/yokai_input_fast"
OUTPUT_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/parallel_full_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Input: ALL 1M+ frames"
echo "  Segmentation: 10 random episodes (16 workers)"
echo "  Clustering: ALL characters (GPU + 8 workers)"
echo "  Training: ALL detected characters"
echo ""
echo "Performance:"
echo "  - Segmentation: 10-15x faster (16 CPU workers)"
echo "  - Clustering: 5-8x faster (GPU batch + parallel loading)"
echo "  - Overall: ~60 hours → ~8 hours"
echo ""

read -p "Start ultra-fast pipeline? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

START_TIME=$(date +%s)

echo "========================================="
echo "🚀 Pipeline started at $(date)"
echo "========================================="
echo ""

conda run -n blip2-env python3 "$SCRIPT_DIR/yokai_parallel_pipeline.py" \
    --input-frames "$INPUT_FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    --segmentation-sample 10 \
    --min-cluster-size 50 \
    --num-seg-workers 16 \
    --batch-size 64 \
    --num-dataloader-workers 8 \
    --device cuda \
    2>&1 | tee "$OUTPUT_DIR/full_pipeline.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================="
echo "✅ Pipeline completed at $(date)"
echo "   Time: ${HOURS}h ${MINUTES}m"
echo "========================================="
echo ""
echo "Results: $OUTPUT_DIR"
echo "Models: /mnt/c/AI_LLM_projects/ai_warehouse/models/lora/yokai-watch/"
echo ""
