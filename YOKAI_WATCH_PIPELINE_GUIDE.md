# Yokai Watch 自動化訓練流程指南

## 📋 概述

這個自動化流程會處理所有 Yokai Watch 的 frames，執行完整的 LoRA 訓練流程：

1. **分層分割** - 將 frames 分離成角色層和背景層
2. **角色聚類** - 自動識別並分組角色
3. **AI 分析** - 分析角色特徵和風格
4. **標註生成** - 自動生成訓練標註
5. **LoRA 訓練** - 訓練每個角色的 LoRA 模型
6. **測試評估** - 生成測試圖片並評估品質

## 🎯 設計特點

### 空間效率設計
- **批次處理**: 一次處理 10 個 episodes，處理完即清理臨時文件
- **漸進式合併**: 每批次的角色聚類會合併到最終輸出
- **最小臨時空間**: 只保留當前批次的分層文件，在 `/tmp` 目錄處理
- **最終保留**: 只保留角色聚類、訓練數據和模型文件

### 自動化設計
- **無人值守**: 可以連續運行處理所有角色
- **錯誤處理**: 單個角色失敗不會影響其他角色
- **進度記錄**: 詳細的日誌和 JSON 結果文件
- **斷點續傳**: 可以跳過已完成的階段

## 📊 數據規模

**當前數據**:
- 📁 Episodes: 214 個季度資料夾
- 🖼️ 總 Frames: 1,042,599 張
- 📍 數據位置: `/home/b0979/yokai_input_fast`

**預估輸出**:
- 👥 角色數量: 預估 50-100+ 個角色 (min_cluster_size=25)
- 💾 每個角色: 25-500+ 張訓練圖片
- 🎓 模型數量: 每個角色產生 1 個 LoRA 模型

## 🚀 使用方式

### 方案 1: 快速測試（推薦先執行）

測試基本流程是否正常，只處理 5 個 episodes 和 2 個角色：

\`\`\`bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora
./scripts/batch/test_yokai_pipeline.sh
\`\`\`

**預計時間**: 約 1-2 小時
**輸出位置**: `/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/quick_test`

### 方案 2: 完整流程（過夜執行）

處理所有數據：

\`\`\`bash
# 1. 啟動 tmux session（重要！避免斷線）
tmux new -s yokai_training

# 2. 進入項目目錄
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora

# 3. 執行完整流程
./scripts/batch/run_yokai_full_pipeline.sh

# 4. 分離 tmux (Ctrl+B 然後按 D)
# 這樣可以關閉終端，流程繼續運行
\`\`\`

**預計時間**:
- 分層分割: ~30-50 小時 (1M+ frames)
- 角色聚類: ~5-10 小時
- 每個角色訓練: ~1-2 小時
- **總計**: 約 2-3 天

### 方案 3: 自訂參數

\`\`\`bash
# 只處理前 50 個 episodes
./scripts/batch/run_yokai_full_pipeline.sh --total-episodes 50

# 只訓練前 10 個角色
./scripts/batch/run_yokai_full_pipeline.sh --max-characters 10

# 組合使用
./scripts/batch/run_yokai_full_pipeline.sh \\
    --total-episodes 50 \\
    --max-characters 10 \\
    --episodes-per-batch 5 \\
    --min-cluster-size 30
\`\`\`

**參數說明**:
- `--total-episodes N`: 只處理前 N 個 episodes（預設：全部）
- `--max-characters N`: 只訓練前 N 個角色（預設：全部）
- `--episodes-per-batch N`: 每批次處理 N 個 episodes（預設：10）
- `--min-cluster-size N`: 角色最小圖片數量（預設：25）

## 📂 輸出結構

\`\`\`
/mnt/c/AI_LLM_projects/ai_warehouse/

├── training_data/yokai-watch/
│   ├── cluster_000/                    # 角色 0 的訓練數據
│   │   ├── *.png                       # 角色圖片
│   │   └── *.txt                       # 對應標註
│   ├── cluster_001/                    # 角色 1
│   └── ...
│
├── models/lora/yokai-watch/
│   ├── cluster_000/
│   │   ├── cluster_000.safetensors     # 最終模型
│   │   └── cluster_000-epoch*.safetensors  # 檢查點
│   ├── cluster_001/
│   └── ...
│
└── outputs/yokai-watch/
    ├── pipeline/                       # 流程輸出
    │   ├── pipeline_log_*.txt         # 詳細日誌
    │   ├── pipeline_results.json      # 結果摘要
    │   └── full_pipeline.log          # 完整輸出
    │
    ├── cluster_000/
    │   ├── ai_analysis/               # AI 分析結果
    │   └── lora_tests/                # 測試生成圖片
    └── ...
\`\`\`

## 🔍 監控進度

### 檢查 tmux session

\`\`\`bash
# 列出所有 sessions
tmux ls

# 重新連接到 session
tmux attach -t yokai_training

# 在 tmux 內部切換視窗
Ctrl+B 然後按 [    # 進入滾動模式，可以上下查看
Q                    # 退出滾動模式
\`\`\`

### 查看實時日誌

\`\`\`bash
# 方法 1: 查看最新的流程日誌
tail -f /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/*/full_pipeline.log

# 方法 2: 查看詳細日誌
tail -f /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/*/pipeline_log_*.txt
\`\`\`

### 查看 GPU 使用情況

\`\`\`bash
# 實時監控
watch -n 1 nvidia-smi

# 或者一次性查看
nvidia-smi
\`\`\`

### 查看進度摘要

\`\`\`bash
# 查看 JSON 結果文件
cat /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/*/pipeline_results.json | jq '.'

# 快速統計
cat /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/*/pipeline_results.json | jq '.processed'
\`\`\`

## ⚠️ 注意事項

### 空間需求

- **臨時空間** (`/tmp`): 約 50-100 GB（批次處理會自動清理）
- **最終輸出**:
  - 訓練數據: 約 20-50 GB（角色圖片 + 標註）
  - 模型文件: 約 5-10 GB（每個模型 ~50-100 MB）
  - 測試圖片: 約 5-10 GB

### 時間安排

建議在週末或過夜執行完整流程：
- **週五晚上啟動** → 週一早上完成
- **使用 tmux** 確保即使斷線也能繼續執行

### 錯誤處理

流程設計為容錯：
- 單個 episode 分割失敗 → 記錄並繼續
- 單個角色訓練失敗 → 記錄並處理下一個
- 可以重新執行，會跳過已完成的聚類階段

## 🔧 進階選項

### 只執行特定階段

如果你已經有聚類結果，可以直接訓練：

\`\`\`python
# 編輯腳本添加參數
python3 scripts/batch/yokai_watch_space_efficient.py \\
    --input-frames /home/b0979/yokai_input_fast \\
    --output-dir /path/to/output \\
    --skip-segmentation \\
    --skip-clustering \\
    --device cuda
\`\`\`

### 調整訓練參數

編輯 `scripts/batch/yokai_watch_space_efficient.py`，找到 `train_character_lora` 函數，可以修改：
- `--max_train_steps`: 訓練步數（預設 2000）
- `--learning_rate`: 學習率（預設 1e-4）
- `--network_dim`: LoRA 維度（預設 32）
- `--train_batch_size`: 批次大小（預設 2）

## 📈 預期結果

執行完成後，你將得到：

1. **50-100+ 個角色的 LoRA 模型**
   - 每個模型可以生成該角色的圖片
   - 不同訓練階段的檢查點可以比較

2. **完整的訓練數據集**
   - 自動標註的角色圖片
   - 可用於後續微調或其他訓練

3. **測試生成的圖片**
   - 每個角色的測試樣本
   - 可以直觀評估模型品質

4. **詳細的分析報告**
   - AI 分析的角色特徵
   - 訓練過程的日誌和統計

## 🎉 開始執行

建議執行順序：

\`\`\`bash
# 1. 先執行快速測試（1-2 小時）
./scripts/batch/test_yokai_pipeline.sh

# 2. 檢查測試結果
ls -lh /mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/quick_test/

# 3. 如果測試成功，啟動完整流程
tmux new -s yokai_training
./scripts/batch/run_yokai_full_pipeline.sh

# 4. 分離 tmux（Ctrl+B, D）然後去休息 :)
\`\`\`

## 📞 問題排查

### 記憶體不足
- 降低 `--episodes-per-batch`（例如改為 5）
- 確保 `/tmp` 有足夠空間

### GPU 錯誤
- 檢查 `nvidia-smi` 確認 GPU 可用
- 嘗試添加 `--device cpu` 使用 CPU（會很慢）

### 訓練失敗
- 檢查 `pipeline_log_*.txt` 查看詳細錯誤
- 某些角色圖片太少會自動跳過（< min_cluster_size）

---

**祝訓練順利！早上見！** 🌅
