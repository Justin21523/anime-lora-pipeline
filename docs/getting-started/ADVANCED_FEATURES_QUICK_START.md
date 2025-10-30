# Yokai Watch Advanced Features - Quick Start Guide

快速開始使用進階訓練功能的指南。

---

## ✅ 已實作功能 (6個核心工具)

### 1. 召喚場景檢測 🌟
**工具**: `yokai_summon_scene_detector.py`

檢測華麗的妖怪召喚動畫場景。

**使用方法**:
```bash
python3 scripts/tools/yokai_summon_scene_detector.py \
    /path/to/episodes \
    --output-dir summon_scenes \
    --extract-mode key \
    --min-score 60.0
```

**功能**:
- 視覺特效檢測（閃光、魔法陣、光束）
- 音訊分析（音效、能量）
- 場景評分（0-100分）
- 三種提取模式：key（關鍵幀）、all（全部）、sample（採樣）

**輸出**:
```
summon_scenes/
├── S1.01/
│   ├── scene_001234/
│   │   ├── scene_001234_frame_001240.png
│   │   └── ...
│   └── ...
└── summon_scenes_metadata.json
```

---

### 2. 動作序列提取 🎬
**工具**: `action_sequence_extractor.py`

提取連續動作序列用於 AnimateDiff motion LoRA 訓練。

**使用方法**:
```bash
python3 scripts/tools/action_sequence_extractor.py \
    /path/to/episodes \
    --output-dir action_sequences \
    --lengths 16 32 64 \
    --format animatediff
```

**功能**:
- 光流運動檢測
- 場景一致性分析
- 動作類型分類（特效、加速、出場等）
- 支持 8/16/32/64 幀序列
- AnimateDiff 格式輸出

**輸出**:
```
action_sequences/
├── S1.01_seq001234_len16_entrance/
│   ├── 0000.png
│   ├── 0001.png
│   ├── ...
│   ├── 0015.png
│   └── metadata.json
└── sequences_metadata.json
```

---

### 3. 特效組織器 ✨
**工具**: `special_effects_organizer.py`

分類和組織特效場景。

**使用方法**:
```bash
python3 scripts/tools/special_effects_organizer.py \
    summon_scenes \
    --output-dir organized_effects \
    --separate-layers
```

**功能**:
- 特效類型分類（召喚、攻擊、變身、魔法陣等）
- 特效/角色層分離
- 純特效提取
- 強度分析

**輸出**:
```
organized_effects/
├── by_type/
│   ├── summon/
│   ├── attack/
│   ├── magic_circle/
│   └── ...
├── pure_effects/
│   ├── summon/
│   └── ...
└── combined/
```

---

### 4. 風格分類器 🎨
**工具**: `yokai_style_classifier.py`

使用 AI 自動分類妖怪風格和屬性。

**使用方法**:
```bash
# 自動分類 + 互動審核
python3 scripts/tools/yokai_style_classifier.py \
    /path/to/character_clusters \
    --output-json yokai_taxonomy.json

# 純自動（不互動）
python3 scripts/tools/yokai_style_classifier.py \
    /path/to/character_clusters \
    --output-json yokai_taxonomy.json \
    --no-interactive
```

**功能**:
- CLIP AI 自動分類
- 多維度分類：
  - 外觀：animal_cat, animal_dog, humanoid, ghost 等
  - 屬性：fire, water, wind, thunder 等
  - 風格：cute, cool, brave, scary 等
  - 體型：quadruped, bipedal, flying 等
- 互動式審核和調整

**輸出**:
```json
{
  "clusters": [
    {
      "cluster_name": "cluster_000",
      "classifications": {
        "appearance": {"animal_cat": 0.85},
        "attribute": {"fire": 0.72},
        "style": {"cute": 0.91}
      }
    }
  ],
  "statistics": {...}
}
```

---

### 5. 多概念 LoRA 準備器 🎯
**工具**: `multi_concept_lora_preparer.py`

準備風格組合訓練（如"所有貓型妖怪"）。

**使用方法**:

首先創建 groups 定義文件 `groups.json`:
```json
[
  {
    "name": "cat_type_yokai",
    "category": "appearance",
    "values": ["animal_cat"]
  },
  {
    "name": "cute_yokai",
    "category": "style",
    "values": ["cute"]
  },
  {
    "name": "fire_attribute",
    "category": "attribute",
    "values": ["fire"]
  }
]
```

然後運行：
```bash
python3 scripts/tools/multi_concept_lora_preparer.py \
    /path/to/character_clusters \
    --taxonomy yokai_taxonomy.json \
    --output-dir multi_concept_training \
    --groups groups.json \
    --training-type concept
```

**功能**:
- 按風格/類型組合多個角色
- 層級觸發詞系統（"yokai, cat-type, char001"）
- 樣本平衡
- 自動參數優化

**輸出**:
```
multi_concept_training/
├── cat_type_yokai/
│   ├── 12_cat_type_yokai/
│   │   ├── cluster_000_image001.png
│   │   ├── cluster_000_image001.txt  # "yokai, cat-type, char000, ..."
│   │   └── ...
│   ├── validation/
│   └── configs/
│       └── cat_type_yokai_config.toml
└── multi_concept_metadata.json
```

**訓練**:
```bash
accelerate launch train_network.py \
    --config_file multi_concept_training/cat_type_yokai/configs/cat_type_yokai_config.toml
```

---

### 6. ControlNet 完整管道 🎮
**工具**: `controlnet_complete_pipeline.py`

一次生成所有 ControlNet 控制圖。

**使用方法**:
```bash
# 生成所有類型
python3 scripts/tools/controlnet_complete_pipeline.py \
    /path/to/character_images \
    --output-dir controlnet_dataset

# 只生成特定類型
python3 scripts/tools/controlnet_complete_pipeline.py \
    /path/to/character_images \
    --output-dir controlnet_dataset \
    --control-types canny depth openpose

# 使用背景層改善深度圖
python3 scripts/tools/controlnet_complete_pipeline.py \
    /path/to/character_images \
    --output-dir controlnet_dataset \
    --background-dir /path/to/backgrounds
```

**功能**:
- **Canny**: 邊緣檢測
- **Depth**: 深度圖生成（動畫優化）
- **OpenPose**: 姿態檢測
- **Lineart**: 線稿提取
- **Segmentation**: 分割遮罩

**輸出**:
```
controlnet_dataset/
├── source/           # 原圖
│   ├── image001.png
│   └── ...
├── canny/           # Canny 邊緣
│   ├── image001.png
│   └── ...
├── depth/           # 深度圖
├── openpose/        # 姿態骨架
├── lineart/         # 線稿
├── segmentation/    # 分割遮罩
├── captions/        # Caption 文字
└── controlnet_metadata.json
```

---

## 🔄 完整工作流程示例

### 工作流程 1: 召喚特效 LoRA

```bash
# 1. 檢測召喚場景
python3 scripts/tools/yokai_summon_scene_detector.py \
    /home/b0979/yokai_input_fast \
    --output-dir summon_scenes \
    --extract-mode sample

# 2. 組織特效
python3 scripts/tools/special_effects_organizer.py \
    summon_scenes \
    --output-dir organized_effects \
    --separate-layers

# 3. 生成 Captions（使用現有工具）
python3 scripts/tools/batch_generate_captions_yokai.py \
    organized_effects/by_type/summon

# 4. 準備訓練（使用現有工具）
python3 scripts/tools/prepare_yokai_lora_training.py \
    organized_effects/by_type/summon \
    --output-dir training_data/summon_effects

# 5. 訓練
accelerate launch train_network.py \
    --config_file training_data/summon_effects/configs/char_000_config.toml
```

---

### 工作流程 2: 風格組合 LoRA（貓型妖怪）

```bash
# 1. 風格分類
python3 scripts/tools/yokai_style_classifier.py \
    character_clusters \
    --output-json yokai_taxonomy.json

# 2. 創建 groups 定義
cat > cat_group.json <<EOF
[
  {
    "name": "cat_type_yokai",
    "category": "appearance",
    "values": ["animal_cat"]
  }
]
EOF

# 3. 準備多概念訓練
python3 scripts/tools/multi_concept_lora_preparer.py \
    character_clusters \
    --taxonomy yokai_taxonomy.json \
    --output-dir cat_lora_training \
    --groups cat_group.json \
    --training-type concept

# 4. 訓練
accelerate launch train_network.py \
    --config_file cat_lora_training/cat_type_yokai/configs/cat_type_yokai_config.toml
```

**結果**: 一個 LoRA 可以生成所有貓型妖怪，使用觸發詞：
- "yokai, cat-type" - 通用貓型風格
- "yokai, cat-type, char000" - 特定妖怪

---

### 工作流程 3: AnimateDiff Motion LoRA

```bash
# 1. 提取動作序列
python3 scripts/tools/action_sequence_extractor.py \
    /home/b0979/yokai_input_fast \
    --output-dir motion_sequences \
    --lengths 16 32 \
    --format animatediff

# 2. 選擇特定動作類型（如出場動畫）
mkdir entrance_sequences
cp -r motion_sequences/*_entrance/ entrance_sequences/

# 3. 訓練 motion LoRA（需要 AnimateDiff 訓練腳本）
# 使用 AnimateDiff 訓練流程...
```

---

### 工作流程 4: ControlNet 訓練數據

```bash
# 1. 生成所有控制圖
python3 scripts/tools/controlnet_complete_pipeline.py \
    character_clusters/cluster_000 \
    --output-dir controlnet_data \
    --background-dir layered_frames/background

# 2. 使用 ControlNet 訓練腳本訓練...
```

---

## 📝 待完成功能

以下功能有完整規格（見 `ADVANCED_TOOLS_SPECIFICATION.md`），但尚未實作：

1. **scene_type_classifier.py** - 場景類型分類（室內/戶外等）
2. **interactive_style_organizer.py** - 視覺化風格組織界面
3. **advanced_pose_extractor.py** - 進階姿態提取（妖怪特殊處理）
4. **anime_depth_generator.py** - 動畫風格深度生成器

如需這些功能，可以參考規格文檔實作，或請求協助。

---

## 💡 使用建議

### 優先使用場景

1. **需要特效 LoRA** → 使用召喚場景檢測 + 特效組織器
2. **需要風格 LoRA** → 使用風格分類器 + 多概念準備器
3. **需要動畫生成** → 使用動作序列提取器 + AnimateDiff
4. **需要精確控制** → 使用 ControlNet 管道

### 組合使用

多個 LoRA 可以組合使用：
```
基礎模型
  + 角色 LoRA (jibanyan)
  + 特效 LoRA (summon_effects)
  + ControlNet Pose
  = 特定姿態的吉胖喵召喚動畫
```

---

## 🔧 依賴安裝

```bash
# 基礎依賴（已有）
conda activate blip2-env

# ControlNet 相關（需要時安裝）
pip install controlnet_aux

# 音訊處理（需要時安裝）
pip install librosa soundfile

# OpenPose（可選）
# 參考：https://github.com/CMU-Perceptual-Computing-Lab/openpose
```

---

## 🆘 故障排除

### CUDA Out of Memory
```bash
# 使用 CPU
--device cpu

# 或減少批次大小
--batch-size 4
```

### CLIP 模型加載失敗
```bash
# 確認網絡連接，模型會自動下載
# 或手動下載到 ~/.cache/huggingface/
```

### 音訊分析失敗
```bash
# 安裝 librosa
pip install librosa soundfile

# 或禁用音訊
--no-audio
```

---

**最後更新**: 2025-10-30
**工具版本**: v1.0
**狀態**: 6/13 核心功能已實作
