# Advanced Tools Quick Reference

快速查閱所有進階工具的參數和用法。

**分類標準版本**: 2025-10 Yokai schema 擴充版
**參照文件**: `docs/YOKAI_SCHEMA_EXTENDED.md`

---

## 📑 目錄

- [召喚場景檢測](#召喚場景檢測)
- [動作序列提取](#動作序列提取)
- [特效組織器](#特效組織器)
- [風格分類器](#風格分類器)
- [多概念準備器](#多概念準備器)
- [ControlNet 管道](#controlnet-管道)
- [整合管道](#整合管道)

---

## 召喚場景檢測

### yokai_summon_scene_detector.py

檢測華麗的妖怪召喚場景。

#### 基本用法

```bash
python3 scripts/tools/yokai_summon_scene_detector.py \
    <episodes_dir> \
    --output-dir <output_dir>
```

#### 完整參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `episodes_dir` | 位置參數 | - | 影片檔案目錄 |
| `--output-dir` | Path | required | 輸出目錄 |
| `--extract-mode` | str | `sample` | 提取模式: `key`/`all`/`sample` |
| `--min-score` | float | `60.0` | 最低場景評分 (0-100) |
| `--device` | str | `cuda` | 處理設備: `cuda`/`cpu` |
| `--batch-size` | int | `8` | 批次大小 |
| `--no-audio` | flag | - | 禁用音訊分析 |

#### 提取模式

- **key**: 只提取關鍵幀（最少圖片，最高品質）
- **sample**: 採樣提取，每 N 幀提取一幀（平衡）
- **all**: 提取所有幀（最多圖片）

#### 評分範圍

- **60-70**: 一般召喚場景
- **70-85**: 華麗召喚場景
- **85-100**: 超華麗召喚場景

#### 輸出結構

```
output_dir/
├── S1.01/
│   ├── scene_001234/
│   │   ├── scene_001234_frame_001240.png
│   │   └── ...
│   └── scene_002456/
└── summon_scenes_metadata.json
```

#### 範例

```bash
# 檢測高品質召喚場景（只提取關鍵幀）
python3 scripts/tools/yokai_summon_scene_detector.py \
    /home/b0979/yokai_input_fast \
    --output-dir summon_scenes_high_quality \
    --extract-mode key \
    --min-score 75.0

# 快速檢測（採樣模式，較低門檻）
python3 scripts/tools/yokai_summon_scene_detector.py \
    /home/b0979/yokai_input_fast \
    --output-dir summon_scenes_all \
    --extract-mode sample \
    --min-score 50.0 \
    --batch-size 16

# CPU 模式（無 GPU）
python3 scripts/tools/yokai_summon_scene_detector.py \
    /home/b0979/yokai_input_fast \
    --output-dir summon_scenes \
    --device cpu \
    --no-audio
```

---

## 動作序列提取

### action_sequence_extractor.py

提取連續動作序列用於 AnimateDiff motion LoRA。

#### 基本用法

```bash
python3 scripts/tools/action_sequence_extractor.py \
    <episodes_dir> \
    --output-dir <output_dir> \
    --lengths 16 32
```

#### 完整參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `episodes_dir` | 位置參數 | - | 影片檔案目錄 |
| `--output-dir` | Path | required | 輸出目錄 |
| `--lengths` | int[] | `[16]` | 序列長度（可多個）: 8/16/32/64 |
| `--format` | str | `animatediff` | 輸出格式: `animatediff`/`standard` |
| `--min-motion` | float | `2.0` | 最小運動強度 |
| `--max-motion` | float | `50.0` | 最大運動強度 |
| `--min-consistency` | float | `0.85` | 最小場景一致性 (0-1) |
| `--device` | str | `cuda` | 處理設備 |

#### 序列長度

- **8**: 超短動作（快速動作）
- **16**: 短動作（出場、轉身）⭐ 推薦
- **32**: 中等動作（攻擊、移動）
- **64**: 長動作（完整召喚序列）

#### 輸出格式

- **animatediff**: `0000.png, 0001.png, ...`（AnimateDiff 標準）
- **standard**: `frame_0000.png, frame_0001.png, ...`

#### 輸出結構

```
output_dir/
├── S1.01_seq001234_len16_entrance/
│   ├── 0000.png
│   ├── 0001.png
│   ├── ...
│   ├── 0015.png
│   └── metadata.json
└── sequences_metadata.json
```

#### 範例

```bash
# 提取多種長度序列
python3 scripts/tools/action_sequence_extractor.py \
    /home/b0979/yokai_input_fast \
    --output-dir motion_sequences \
    --lengths 16 32 64 \
    --format animatediff

# 只提取高運動場景
python3 scripts/tools/action_sequence_extractor.py \
    /home/b0979/yokai_input_fast \
    --output-dir high_motion_sequences \
    --lengths 16 \
    --min-motion 10.0 \
    --max-motion 40.0

# 嚴格場景一致性
python3 scripts/tools/action_sequence_extractor.py \
    /home/b0979/yokai_input_fast \
    --output-dir consistent_sequences \
    --lengths 32 \
    --min-consistency 0.95
```

---

## 特效組織器

### special_effects_organizer.py

分類和組織特效場景。

#### 基本用法

```bash
python3 scripts/tools/special_effects_organizer.py \
    <input_dir> \
    --output-dir <output_dir>
```

#### 完整參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `input_dir` | 位置參數 | - | 輸入目錄（如召喚場景） |
| `--output-dir` | Path | required | 輸出目錄 |
| `--separate-layers` | flag | - | 分離特效和角色層 |
| `--intensity-threshold` | float | `0.3` | 特效強度門檻 (0-1) |
| `--device` | str | `cuda` | 處理設備 |

#### 輸出結構

```
output_dir/
├── by_type/
│   ├── summon/           # 召喚特效
│   ├── attack/           # 攻擊特效
│   ├── magic_circle/     # 魔法陣
│   ├── transformation/   # 變身特效
│   └── ambient/          # 環境特效
├── pure_effects/         # 純特效（移除角色）
│   ├── summon/
│   └── ...
└── combined/             # 特效+角色
```

#### 特效類型

- **summon**: 召喚動畫
- **attack**: 攻擊特效
- **transformation**: 變身/進化
- **magic_circle**: 魔法陣
- **ambient**: 環境光效

#### 範例

```bash
# 基本組織
python3 scripts/tools/special_effects_organizer.py \
    summon_scenes \
    --output-dir organized_effects

# 分離特效層（用於純特效訓練）
python3 scripts/tools/special_effects_organizer.py \
    summon_scenes \
    --output-dir organized_effects \
    --separate-layers

# 只保留高強度特效
python3 scripts/tools/special_effects_organizer.py \
    summon_scenes \
    --output-dir high_intensity_effects \
    --intensity-threshold 0.6 \
    --separate-layers
```

---

## 風格分類器

### yokai_style_classifier.py

使用 AI 自動分類妖怪風格和屬性。

#### 基本用法

```bash
python3 scripts/tools/yokai_style_classifier.py \
    <clusters_dir> \
    --output-json <taxonomy.json>
```

#### 完整參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `clusters_dir` | 位置參數 | - | 角色聚類目錄 |
| `--output-json` | Path | required | 輸出分類文件 |
| `--threshold` | float | `0.3` | 分類信心門檻 (0-1) |
| `--sample-size` | int | `10` | 每個聚類採樣圖片數 |
| `--no-interactive` | flag | - | 禁用互動審核 |
| `--device` | str | `cuda` | 處理設備 |

#### 分類維度

1. **appearance** (外觀):
   - animal_cat, animal_dog, animal_bird, animal_dragon
   - object_food, object_tool, object_toy
   - humanoid, ghost, abstract

2. **attribute** (屬性):
   - fire, water, wind, thunder, earth, ice, light, dark, nature

3. **style** (風格):
   - cute, cool, brave, scary, funny, mysterious, elegant

4. **body_type** (體型):
   - quadruped, bipedal, flying, floating, multi_limbed

#### 互動模式

```
[1/128] cluster_000
AI Classification:
  appearance:
    - animal_cat: 0.85
  attribute:
    - fire: 0.68
  style:
    - cute: 0.91

Action (a/m/s/q):
  a - Accept        接受 AI 分類
  m - Modify        修改分類
  s - Skip          跳過此聚類
  q - Quit          退出審核
```

#### 範例

```bash
# 自動分類 + 互動審核（推薦）
python3 scripts/tools/yokai_style_classifier.py \
    character_clusters \
    --output-json yokai_taxonomy.json \
    --threshold 0.3 \
    --sample-size 10

# 純自動（批次處理）
python3 scripts/tools/yokai_style_classifier.py \
    character_clusters \
    --output-json yokai_taxonomy.json \
    --no-interactive

# 嚴格分類（高信心門檻）
python3 scripts/tools/yokai_style_classifier.py \
    character_clusters \
    --output-json yokai_taxonomy_strict.json \
    --threshold 0.5 \
    --sample-size 15

# CPU 模式
python3 scripts/tools/yokai_style_classifier.py \
    character_clusters \
    --output-json yokai_taxonomy.json \
    --device cpu \
    --no-interactive
```

---

## 多概念準備器

### multi_concept_lora_preparer.py

準備風格組合訓練（如「所有貓型妖怪」）。

#### 基本用法

```bash
python3 scripts/tools/multi_concept_lora_preparer.py \
    <clusters_dir> \
    --taxonomy <taxonomy.json> \
    --output-dir <output_dir> \
    --groups <groups.json>
```

#### 完整參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `clusters_dir` | 位置參數 | - | 角色聚類目錄 |
| `--taxonomy` | Path | required | 風格分類文件 |
| `--output-dir` | Path | required | 輸出目錄 |
| `--groups` | Path | required | 組定義文件 |
| `--training-type` | str | `concept` | 訓練類型: `concept`/`style` |
| `--target-samples` | int | `40` | 目標樣本數/角色 |
| `--max-samples` | int | `80` | 最大樣本數/角色 |
| `--hierarchical` | flag | - | 使用階層觸發詞 |
| `--device` | str | `cuda` | 處理設備 |

#### 組定義格式 (groups.json)

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
  },
  {
    "name": "cute_cat_fire",
    "category": "multi",
    "filters": {
      "appearance": ["animal_cat"],
      "style": ["cute"],
      "attribute": ["fire"]
    }
  }
]
```

#### 訓練類型

- **concept**: 學習多個角色的共同概念
  - 高維度（保留個體特徵）
  - 階層觸發詞

- **style**: 純風格學習
  - 低維度（聚焦共同風格）
  - 更高學習率

#### 觸發詞系統

**階層式**（推薦）:
```
"yokai, cat-type, char000"
Level 1: yokai
Level 2: yokai, cat-type
Level 3: yokai, cat-type, char000
```

**扁平式**:
```
"cat_type_yokai_char000"
```

#### 輸出結構

```
output_dir/
├── cat_type_yokai/
│   ├── 12_cat_type_yokai/
│   │   ├── cluster_000_img001.png
│   │   ├── cluster_000_img001.txt   # "yokai, cat-type, char000, ..."
│   │   └── ...
│   ├── validation/
│   └── configs/
│       └── cat_type_yokai_config.toml
└── multi_concept_metadata.json
```

#### 範例

```bash
# 基本多概念準備
python3 scripts/tools/multi_concept_lora_preparer.py \
    character_clusters \
    --taxonomy yokai_taxonomy.json \
    --output-dir multi_concept_training \
    --groups groups.json \
    --training-type concept

# 風格訓練（聚焦共同特徵）
python3 scripts/tools/multi_concept_lora_preparer.py \
    character_clusters \
    --taxonomy yokai_taxonomy.json \
    --output-dir style_training \
    --groups style_groups.json \
    --training-type style

# 自定義樣本數
python3 scripts/tools/multi_concept_lora_preparer.py \
    character_clusters \
    --taxonomy yokai_taxonomy.json \
    --output-dir balanced_training \
    --groups groups.json \
    --target-samples 50 \
    --max-samples 100

# 使用扁平觸發詞
python3 scripts/tools/multi_concept_lora_preparer.py \
    character_clusters \
    --taxonomy yokai_taxonomy.json \
    --output-dir flat_trigger_training \
    --groups groups.json
    # 不加 --hierarchical
```

---

## ControlNet 管道

### controlnet_complete_pipeline.py

一次生成所有 ControlNet 控制圖。

#### 基本用法

```bash
python3 scripts/tools/controlnet_complete_pipeline.py \
    <input_dir> \
    --output-dir <output_dir>
```

#### 完整參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `input_dir` | 位置參數 | - | 輸入圖片目錄 |
| `--output-dir` | Path | required | 輸出目錄 |
| `--control-types` | str[] | all | 控制類型（可多個） |
| `--background-dir` | Path | - | 背景目錄（改善深度圖） |
| `--device` | str | `cuda` | 處理設備 |

#### 控制類型

- **canny**: Canny 邊緣檢測
- **depth**: 深度圖生成
- **openpose**: OpenPose 姿態檢測
- **lineart**: 線稿提取
- **segmentation**: 分割遮罩

#### 輸出結構

```
output_dir/
├── source/              # 原圖
├── canny/               # Canny 邊緣
├── depth/               # 深度圖
├── openpose/            # 姿態骨架
├── lineart/             # 線稿
├── segmentation/        # 分割遮罩
├── captions/            # Caption 文字
└── controlnet_metadata.json
```

#### 範例

```bash
# 生成所有類型
python3 scripts/tools/controlnet_complete_pipeline.py \
    character_clusters/cluster_000 \
    --output-dir controlnet_all

# 只生成特定類型
python3 scripts/tools/controlnet_complete_pipeline.py \
    character_clusters/cluster_000 \
    --output-dir controlnet_pose_depth \
    --control-types openpose depth

# 使用背景層改善深度
python3 scripts/tools/controlnet_complete_pipeline.py \
    character_clusters/cluster_000 \
    --output-dir controlnet_with_bg \
    --control-types depth \
    --background-dir layered_frames/background

# 批次處理所有聚類
for cluster in character_clusters/cluster_*; do
    cluster_name=$(basename "$cluster")
    python3 scripts/tools/controlnet_complete_pipeline.py \
        "$cluster" \
        --output-dir "controlnet_datasets/$cluster_name" \
        --control-types canny depth openpose
done
```

---

## 整合管道

### yokai_advanced_training_pipeline.sh

整合所有進階功能的完整管道。

#### 基本用法

```bash
./scripts/batch/yokai_advanced_training_pipeline.sh
```

#### 環境變數配置

```bash
# 輸入目錄
export EPISODES_DIR="/home/b0979/yokai_input_fast"
export CHARACTER_CLUSTERS_DIR="/path/to/character_clusters"
export LAYERED_FRAMES_DIR="/path/to/layered_frames"

# 輸出目錄
export OUTPUT_BASE="/path/to/advanced_output"

# 功能開關
export ENABLE_SUMMON_DETECTION="true"
export ENABLE_ACTION_SEQUENCES="true"
export ENABLE_EFFECTS_ORGANIZATION="true"
export ENABLE_STYLE_CLASSIFICATION="true"
export ENABLE_MULTI_CONCEPT="true"
export ENABLE_CONTROLNET="true"

# 處理參數
export SUMMON_MIN_SCORE="60.0"
export SUMMON_EXTRACT_MODE="sample"
export ACTION_LENGTHS="16 32"
export STYLE_THRESHOLD="0.3"
export STYLE_INTERACTIVE="true"
export CONTROLNET_TYPES="canny depth openpose"

# 設備
export DEVICE="cuda"
export CONDA_ENV="blip2-env"

# 運行管道
./scripts/batch/yokai_advanced_training_pipeline.sh
```

#### 處理階段

1. **階段 1**: 召喚場景檢測
2. **階段 2**: 動作序列提取
3. **階段 3**: 特效組織
4. **階段 4**: 風格分類
5. **階段 5**: 多概念準備
6. **階段 6**: ControlNet 預處理

#### 輸出結構

```
OUTPUT_BASE/
├── summon_scenes/
├── action_sequences/
├── organized_effects/
├── yokai_taxonomy.json
├── concept_groups.json
├── multi_concept_training/
├── controlnet_datasets/
└── pipeline_summary.txt
```

#### 範例

```bash
# 運行所有階段
./scripts/batch/yokai_advanced_training_pipeline.sh

# 只運行特定階段
export ENABLE_SUMMON_DETECTION="true"
export ENABLE_ACTION_SEQUENCES="false"
export ENABLE_EFFECTS_ORGANIZATION="false"
export ENABLE_STYLE_CLASSIFICATION="false"
export ENABLE_MULTI_CONCEPT="false"
export ENABLE_CONTROLNET="false"
./scripts/batch/yokai_advanced_training_pipeline.sh

# 高品質模式
export SUMMON_MIN_SCORE="75.0"
export SUMMON_EXTRACT_MODE="key"
export STYLE_THRESHOLD="0.4"
export STYLE_SAMPLE_SIZE="15"
./scripts/batch/yokai_advanced_training_pipeline.sh

# 快速模式
export SUMMON_MIN_SCORE="50.0"
export SUMMON_EXTRACT_MODE="sample"
export ACTION_LENGTHS="16"
export STYLE_INTERACTIVE="false"
export CONTROLNET_TYPES="canny depth"
./scripts/batch/yokai_advanced_training_pipeline.sh
```

---

## 🔗 工作流程整合

### 工作流程 1: 召喚特效 LoRA

```bash
# 1. 檢測
python3 scripts/tools/yokai_summon_scene_detector.py \
    /home/b0979/yokai_input_fast \
    --output-dir summon_scenes \
    --extract-mode sample

# 2. 組織
python3 scripts/tools/special_effects_organizer.py \
    summon_scenes \
    --output-dir organized_effects \
    --separate-layers

# 3. Caption + 準備（使用現有工具）
python3 scripts/tools/batch_generate_captions_yokai.py organized_effects/by_type/summon
python3 scripts/tools/prepare_yokai_lora_training.py organized_effects/by_type/summon \
    --output-dir training_data/summon_effects

# 4. 訓練
accelerate launch train_network.py \
    --config_file training_data/summon_effects/configs/summon_effects_config.toml
```

### 工作流程 2: 風格組合 LoRA

```bash
# 1. 分類
python3 scripts/tools/yokai_style_classifier.py \
    character_clusters \
    --output-json yokai_taxonomy.json

# 2. 定義組
cat > cat_group.json <<EOF
[{"name": "cat_type_yokai", "category": "appearance", "values": ["animal_cat"]}]
EOF

# 3. 準備
python3 scripts/tools/multi_concept_lora_preparer.py \
    character_clusters \
    --taxonomy yokai_taxonomy.json \
    --output-dir cat_lora_training \
    --groups cat_group.json

# 4. 訓練
accelerate launch train_network.py \
    --config_file cat_lora_training/cat_type_yokai/configs/cat_type_yokai_config.toml
```

### 工作流程 3: 完整管道

```bash
# 一次運行所有階段
./scripts/batch/yokai_advanced_training_pipeline.sh

# 查看總結
cat /path/to/advanced_output/pipeline_summary.txt
```

---

## 🛠️ 常用參數組合

### 高品質設定

```bash
--min-score 75.0
--extract-mode key
--threshold 0.4
--sample-size 15
--target-samples 50
```

### 平衡設定（推薦）

```bash
--min-score 60.0
--extract-mode sample
--threshold 0.3
--sample-size 10
--target-samples 40
```

### 快速測試設定

```bash
--min-score 50.0
--extract-mode sample
--threshold 0.2
--sample-size 5
--target-samples 20
--no-interactive
```

### 低資源設定

```bash
--device cpu
--batch-size 4
--sample-size 5
--no-audio
```

---

## 📊 輸出文件參考

### summon_scenes_metadata.json

```json
{
  "total_scenes": 45,
  "episodes_processed": 12,
  "scenes": [
    {
      "scene_id": "scene_001234",
      "episode": "S1.01",
      "start_frame": 1234,
      "end_frame": 1256,
      "score": 85.5,
      "effects": {
        "has_flash": true,
        "has_magic_circle": true,
        "num_circles": 2,
        "radial_score": 0.8
      }
    }
  ]
}
```

### yokai_taxonomy.json

```json
{
  "total_clusters": 128,
  "clusters": [
    {
      "cluster_name": "cluster_000",
      "num_samples": 10,
      "classifications": {
        "appearance": {"animal_cat": 0.85},
        "attribute": {"fire": 0.68},
        "style": {"cute": 0.91},
        "body_type": {"quadruped": 0.72}
      }
    }
  ],
  "statistics": {
    "by_appearance": {"animal_cat": 15, "humanoid": 8},
    "by_attribute": {"fire": 12, "water": 10}
  }
}
```

### multi_concept_metadata.json

```json
{
  "groups": [
    {
      "name": "cat_type_yokai",
      "num_clusters": 15,
      "total_samples": 600,
      "trigger_words": "yokai, cat-type",
      "training_params": {
        "network_dim": 48,
        "network_alpha": 24,
        "max_train_epochs": 18
      }
    }
  ]
}
```

---

## 🔍 故障排除

### 記憶體不足

```bash
--device cpu                    # 使用 CPU
--batch-size 4                  # 減少批次
--sample-size 5                 # 減少樣本
```

### CLIP 模型加載失敗

```bash
# 檢查網絡連接
# 模型會自動下載到 ~/.cache/huggingface/

# 或手動下載
huggingface-cli download openai/clip-vit-base-patch32
```

### 音訊處理失敗

```bash
# 安裝依賴
pip install librosa soundfile

# 或禁用音訊
--no-audio
```

### 互動模式無法使用

```bash
# 批次模式
--no-interactive
```

---

## 📚 相關文檔

- **使用指南**: `USAGE_GUIDE.md` - 基礎訓練流程
- **工具參考**: `TOOLS_QUICK_REFERENCE.md` - 基礎工具參考
- **進階快速開始**: `ADVANCED_FEATURES_QUICK_START.md` - 進階功能快速開始
- **進階訓練指南**: `YOKAI_ADVANCED_TRAINING_GUIDE.md` - 完整訓練教程
- **工具規格**: `ADVANCED_TOOLS_SPECIFICATION.md` - 詳細規格文檔

---

**最後更新**: 2025-10-30
**版本**: v1.0
**狀態**: 6/13 核心工具已實作並文檔化
