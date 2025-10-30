#!/bin/bash
#
# Yokai-Watch & Inazuma-Eleven Processing in Tmux
# 使用 tmux 確保長時間運行任務不會被中斷
#

set -e

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Anime Processing Tmux Manager${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 檢查 tmux 是否已安裝
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}✗ tmux not installed${NC}"
    echo "Please install tmux: sudo apt-get install tmux"
    exit 1
fi

echo -e "${GREEN}✓ tmux is installed${NC}"
echo ""

# 配置
SESSION_NAME="anime_processing"
CONDA_ENV="blip2-env"
BASE_DIR="/mnt/c/AI_LLM_projects/inazuma-eleven-lora"
OUTPUT_BASE="/mnt/c/AI_LLM_projects/ai_warehouse/outputs"

# 檢查會話是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo -e "${YELLOW}⚠ Session '$SESSION_NAME' already exists${NC}"
    echo ""
    echo "Options:"
    echo "  1) Attach to existing session"
    echo "  2) Kill and recreate session"
    echo "  3) Exit"
    echo ""
    read -p "Choose [1-3]: " choice

    case $choice in
        1)
            echo -e "${GREEN}Attaching to existing session...${NC}"
            tmux attach-session -t $SESSION_NAME
            exit 0
            ;;
        2)
            echo -e "${YELLOW}Killing existing session...${NC}"
            tmux kill-session -t $SESSION_NAME
            ;;
        3)
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
fi

# 創建新的 tmux 會話
echo -e "${GREEN}Creating tmux session: $SESSION_NAME${NC}"
tmux new-session -d -s $SESSION_NAME -n "main"

# 窗口 0: 主控制台
tmux send-keys -t $SESSION_NAME:0 "cd $BASE_DIR" C-m
tmux send-keys -t $SESSION_NAME:0 "echo '=== Anime Processing Control Panel ==='" C-m
tmux send-keys -t $SESSION_NAME:0 "echo 'Use Ctrl+B then number to switch windows:'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo '  0: Main control (this)'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo '  1: Yokai-Watch processing'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo '  2: Inazuma-Eleven features'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo '  3: Monitoring'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:0 "echo 'Detach: Ctrl+B then D'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo 'Reattach: tmux attach -t $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo ''" C-m

# 窗口 1: Yokai-Watch 完整處理
tmux new-window -t $SESSION_NAME:1 -n "yokai-watch"
tmux send-keys -t $SESSION_NAME:1 "cd $BASE_DIR" C-m
tmux send-keys -t $SESSION_NAME:1 "echo '=== Starting Yokai-Watch Advanced Processing ==='" C-m
tmux send-keys -t $SESSION_NAME:1 "echo 'Processing 1,042,599 frames (219 episodes)...'" C-m
tmux send-keys -t $SESSION_NAME:1 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:1 "conda run -n $CONDA_ENV python3 scripts/tools/yokai_advanced_pipeline.py --device cuda --resume 2>&1 | tee $OUTPUT_BASE/yokai-watch/full_processing.log" C-m

# 窗口 2: Inazuma-Eleven 特徵提取
tmux new-window -t $SESSION_NAME:2 -n "inazuma-features"
tmux send-keys -t $SESSION_NAME:2 "cd $BASE_DIR" C-m
tmux send-keys -t $SESSION_NAME:2 "echo '=== Starting Inazuma-Eleven Feature Extraction ==='" C-m
tmux send-keys -t $SESSION_NAME:2 "echo 'Extracting features from 16,929 frames...'" C-m
tmux send-keys -t $SESSION_NAME:2 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:2 "conda run -n $CONDA_ENV python3 scripts/tools/interactive_character_sorter.py extract /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/layered_frames -o $OUTPUT_BASE/inazuma-eleven_character_features.npz --device cuda" C-m

# 窗口 3: 監控窗口
tmux new-window -t $SESSION_NAME:3 -n "monitoring"
tmux send-keys -t $SESSION_NAME:3 "cd $OUTPUT_BASE" C-m
tmux send-keys -t $SESSION_NAME:3 "echo '=== Processing Monitor ==='" C-m
tmux send-keys -t $SESSION_NAME:3 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:3 "echo 'Available commands:'" C-m
tmux send-keys -t $SESSION_NAME:3 "echo '  tail -f yokai-watch/full_processing.log       # View Yokai-Watch log'" C-m
tmux send-keys -t $SESSION_NAME:3 "echo '  ls yokai-watch/multi_layer_segmentation/     # Check progress'" C-m
tmux send-keys -t $SESSION_NAME:3 "echo '  nvidia-smi                                    # Check GPU usage'" C-m
tmux send-keys -t $SESSION_NAME:3 "echo '  htop                                          # Check CPU/RAM'" C-m
tmux send-keys -t $SESSION_NAME:3 "echo ''" C-m

# 回到主窗口
tmux select-window -t $SESSION_NAME:0

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Tmux session created successfully${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Session: ${BLUE}$SESSION_NAME${NC}"
echo ""
echo -e "${YELLOW}Windows:${NC}"
echo "  0: Main control panel"
echo "  1: Yokai-Watch processing (RUNNING)"
echo "  2: Inazuma-Eleven features (RUNNING)"
echo "  3: Monitoring tools"
echo ""
echo -e "${YELLOW}Usage:${NC}"
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Detach:  Press Ctrl+B then D"
echo "  Switch:  Press Ctrl+B then window number (0-3)"
echo "  Kill:    tmux kill-session -t $SESSION_NAME"
echo ""
echo -e "${GREEN}Attaching to session in 3 seconds...${NC}"
echo "(Press Ctrl+C to cancel)"
sleep 3

# 附加到會話
tmux attach-session -t $SESSION_NAME
