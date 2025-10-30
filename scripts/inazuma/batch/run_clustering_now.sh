#!/bin/bash
# Quick clustering execution script
set -e

cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora

echo "Starting character clustering with hard links..."
echo "Output: /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/character_clusters"
echo ""

conda run -n blip2-env python3 scripts/tools/character_clustering.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/layered_frames \
  --output-dir /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/character_clusters \
  --min-cluster-size 25 \
  --device cuda

echo ""
echo "Clustering complete!"
