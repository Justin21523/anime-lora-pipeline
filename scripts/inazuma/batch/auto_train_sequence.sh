#!/bin/bash
# Automated Sequential LoRA Training
# Trains multiple datasets one after another automatically

set -e  # Exit on error

SDSCRIPTS_DIR="/mnt/c/AI_LLM_projects/sd-scripts"
CONFIG_DIR="/mnt/c/AI_LLM_projects/inazuma-eleven-lora/configs"
LOG_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/outputs/lora_training"

echo "=================================="
echo "🚀 Automated LoRA Training Pipeline"
echo "=================================="
echo ""
echo "Training Schedule:"
echo "  1. Diverse Dataset (SD1.5) - ~3-4 hours"
echo "  2. Battle Dataset (SD1.5) - ~2-3 hours"
echo ""
echo "Total estimated time: 5-7 hours"
echo "=================================="
echo ""

# Function to check if training completed successfully
check_training_success() {
    local log_file=$1
    if grep -q "model saved" "$log_file" && ! grep -q "Error\|Failed\|OOM" "$log_file"; then
        return 0
    else
        return 1
    fi
}

# Training 1: Diverse Dataset
echo "📊 [1/2] Starting Diverse Dataset Training..."
echo "Time: $(date)"
echo ""

cd "$SDSCRIPTS_DIR"
python train_network.py \
    --config_file="$CONFIG_DIR/train_diverse_sd15_updated.toml" \
    --dataset_config="$CONFIG_DIR/dataset_diverse_sd15.toml" \
    2>&1 | tee "$LOG_DIR/train_diverse_sd15.log"

if check_training_success "$LOG_DIR/train_diverse_sd15.log"; then
    echo "✅ Diverse Dataset Training Complete!"
    echo ""
else
    echo "❌ Diverse Dataset Training Failed!"
    echo "Check log: $LOG_DIR/train_diverse_sd15.log"
    exit 1
fi

# Wait a bit for GPU cooldown
echo "⏳ GPU cooldown (30s)..."
sleep 30

# Training 2: Battle Dataset
echo "📊 [2/2] Starting Battle Dataset Training..."
echo "Time: $(date)"
echo ""

python train_network.py \
    --config_file="$CONFIG_DIR/train_battle_sd15.toml" \
    --dataset_config="$CONFIG_DIR/dataset_battle_sd15.toml" \
    2>&1 | tee "$LOG_DIR/train_battle_sd15.log"

if check_training_success "$LOG_DIR/train_battle_sd15.log"; then
    echo "✅ Battle Dataset Training Complete!"
    echo ""
else
    echo "❌ Battle Dataset Training Failed!"
    echo "Check log: $LOG_DIR/train_battle_sd15.log"
    exit 1
fi

# Final summary
echo ""
echo "=================================="
echo "🎉 All Training Complete!"
echo "=================================="
echo "Time: $(date)"
echo ""
echo "Output LoRAs:"
echo "  • Diverse: $LOG_DIR/../inazuma_diverse_sd15/"
echo "  • Battle: $LOG_DIR/../inazuma_battle_sd15/"
echo ""
echo "Logs saved to: $LOG_DIR"
echo "=================================="
