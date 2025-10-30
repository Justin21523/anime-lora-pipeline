#!/bin/bash
# Full Yokai Watch Pipeline
# Process all episodes and train all characters
# Designed for overnight/long-running execution

set -e  # Exit on error

echo "========================================="
echo "Yokai Watch Complete Pipeline"
echo "========================================="
echo ""

# Check if running in tmux
if [ -z "$TMUX" ]; then
    echo "⚠️  Warning: Not running in tmux session"
    echo "   It's recommended to run this in tmux for long-running tasks"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Start a tmux session with: tmux new -s yokai_training"
        exit 1
    fi
fi

# Paths
INPUT_FRAMES="/home/b0979/yokai_input_fast"
OUTPUT_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/full_pipeline_$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="/mnt/c/AI_LLM_projects/inazuma-eleven-lora/scripts/batch"

# Configuration
EPISODES_PER_BATCH=10  # Process 10 episodes at a time
MIN_CLUSTER_SIZE=25    # Minimum 25 images per character
TOTAL_EPISODES=""      # Empty = process all episodes
MAX_CHARACTERS=""      # Empty = train all characters

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes-per-batch)
            EPISODES_PER_BATCH="$2"
            shift 2
            ;;
        --total-episodes)
            TOTAL_EPISODES="$2"
            shift 2
            ;;
        --max-characters)
            MAX_CHARACTERS="$2"
            shift 2
            ;;
        --min-cluster-size)
            MIN_CLUSTER_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--episodes-per-batch N] [--total-episodes N] [--max-characters N] [--min-cluster-size N]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD=(
    conda run -n blip2-env python3 "$SCRIPT_DIR/yokai_watch_space_efficient.py"
    --input-frames "$INPUT_FRAMES"
    --output-dir "$OUTPUT_DIR"
    --episodes-per-batch "$EPISODES_PER_BATCH"
    --min-cluster-size "$MIN_CLUSTER_SIZE"
    --device cuda
)

# Add optional parameters
if [ -n "$TOTAL_EPISODES" ]; then
    CMD+=(--total-episodes "$TOTAL_EPISODES")
fi

if [ -n "$MAX_CHARACTERS" ]; then
    CMD+=(--max-characters "$MAX_CHARACTERS")
fi

# Display configuration
echo "Configuration:"
echo "  Input Frames: $INPUT_FRAMES"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Episodes per Batch: $EPISODES_PER_BATCH"
echo "  Min Cluster Size: $MIN_CLUSTER_SIZE"
if [ -n "$TOTAL_EPISODES" ]; then
    echo "  Total Episodes: $TOTAL_EPISODES"
else
    echo "  Total Episodes: ALL"
fi
if [ -n "$MAX_CHARACTERS" ]; then
    echo "  Max Characters: $MAX_CHARACTERS"
else
    echo "  Max Characters: ALL"
fi
echo ""

# Confirm start
echo "This will process Yokai Watch data and may take many hours."
echo "Results will be saved to: $OUTPUT_DIR"
echo ""
read -p "Start pipeline? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Save start time
START_TIME=$(date +%s)
echo "========================================="
echo "🚀 Pipeline started at $(date)"
echo "========================================="
echo ""

# Run pipeline and save output to log
LOG_FILE="$OUTPUT_DIR/full_pipeline.log"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

# Calculate elapsed time
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
echo "  Output Directory: $OUTPUT_DIR"
echo "  Pipeline Log: $LOG_FILE"
echo "  Results JSON: $OUTPUT_DIR/pipeline_results.json"
echo "  Detailed Log: $OUTPUT_DIR/pipeline_log_*.txt"
echo ""
echo "Models saved to: /mnt/c/AI_LLM_projects/ai_warehouse/models/lora/yokai-watch/"
echo "Training data: /mnt/c/AI_LLM_projects/ai_warehouse/training_data/yokai-watch/"
echo ""
