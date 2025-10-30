#!/bin/bash
# Character Clustering with Hard Links
# Project: Inazuma Eleven LoRA Training
# Location: scripts/batch/run_character_clustering.sh

set -e  # Exit on error

PROJECT_ROOT="/mnt/c/AI_LLM_projects/inazuma-eleven-lora"
WAREHOUSE_ROOT="/mnt/c/AI_LLM_projects/ai_warehouse"

# Source data location
SOURCE_DATA="${WAREHOUSE_ROOT}/cache/inazuma-eleven/layered_frames"

# Output location (will use hard links automatically)
OUTPUT_DIR="${WAREHOUSE_ROOT}/training_data/inazuma-eleven/character_clusters"

# Log file
LOG_DIR="${WAREHOUSE_ROOT}/outputs/character_clustering/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/clustering_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Character Clustering - Inazuma Eleven"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Source:     ${SOURCE_DATA}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Log:        ${LOG_FILE}"
echo "  Min Size:   25 images per cluster"
echo "  Device:     CUDA (GPU)"
echo "  Link Type:  Hard links (Windows-compatible)"
echo ""
echo "=========================================="
echo ""

cd "${PROJECT_ROOT}"

# Run clustering
conda run -n blip2-env python3 scripts/tools/character_clustering.py \
  "${SOURCE_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --min-cluster-size 25 \
  --device cuda \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Clustering completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  ${OUTPUT_DIR}"
    echo ""
    echo "Log file:"
    echo "  ${LOG_FILE}"
else
    echo "✗ Clustering failed with exit code: $EXIT_CODE"
    echo "Check log file for details:"
    echo "  ${LOG_FILE}"
fi
echo "=========================================="

exit $EXIT_CODE
