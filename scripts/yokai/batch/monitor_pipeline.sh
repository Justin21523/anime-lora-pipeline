#!/bin/bash
# Monitor Yokai Watch Pipeline Progress

PIPELINE_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch"

echo "========================================="
echo "Yokai Watch Pipeline Monitor"
echo "========================================="
echo ""

# Find latest pipeline directory
LATEST_DIR=$(ls -td $PIPELINE_DIR/*/ 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No pipeline runs found in $PIPELINE_DIR"
    exit 1
fi

echo "Monitoring: $LATEST_DIR"
echo ""

# Show latest log file
LOG_FILE=$(ls -t "$LATEST_DIR"pipeline_log_*.txt 2>/dev/null | head -1)

if [ -f "$LOG_FILE" ]; then
    echo "--- Last 30 lines of log ---"
    tail -30 "$LOG_FILE"
    echo ""
fi

# Show results JSON if exists
RESULTS_JSON="$LATEST_DIR/pipeline_results.json"
if [ -f "$RESULTS_JSON" ]; then
    echo "--- Progress Summary ---"
    cat "$RESULTS_JSON" | jq -r '"Total Characters: \(.total_characters // 0)\nProcessed: \(.processed // 0)\nLast Update: \(.pipeline_start // "N/A")"'
    echo ""
fi

# Show temp processing status
echo "--- Temporary Processing ---"
if [ -d "/tmp/yokai_processing/layered_frames/character" ]; then
    CHAR_COUNT=$(find /tmp/yokai_processing/layered_frames/character -name "*.png" 2>/dev/null | wc -l)
    echo "Character layers processed: $CHAR_COUNT"
fi

if [ -d "/tmp/yokai_processing/clusters" ]; then
    CLUSTER_COUNT=$(find /tmp/yokai_processing/clusters -type d -name "cluster_*" 2>/dev/null | wc -l)
    echo "Clusters found: $CLUSTER_COUNT"
fi

# Show final outputs
echo ""
echo "--- Final Outputs ---"
TRAINING_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/training_data/yokai-watch"
if [ -d "$TRAINING_DIR" ]; then
    FINAL_CLUSTERS=$(find "$TRAINING_DIR" -type d -name "cluster_*" 2>/dev/null | wc -l)
    echo "Final character clusters: $FINAL_CLUSTERS"
fi

MODEL_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/yokai-watch"
if [ -d "$MODEL_DIR" ]; then
    MODEL_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
    echo "LoRA models trained: $MODEL_COUNT"
fi

echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "========================================="
echo "To watch log in real-time:"
echo "tail -f \"$LOG_FILE\""
echo "========================================="
