# Yokai Watch Advanced Training Guide

完整的進階訓練指南，涵蓋特效、風格、動作和 ControlNet 訓練。

**分類標準版本**: 2025-10 Yokai schema 擴充版
**參照文件**: `docs/YOKAI_SCHEMA_EXTENDED.md`
**來源**: 官方妖怪手錶遊戲/動畫地點、Fandom Wiki 分類

---

## 目錄

1. [特殊場景訓練](#特殊場景訓練)
   - [召喚特效 LoRA](#召喚特效-lora)
   - [攻擊特效 LoRA](#攻擊特效-lora)
   - [動作序列 (AnimateDiff)](#動作序列-animatediff)
2. [風格 LoRA 訓練](#風格-lora-訓練)
   - [概念理解](#概念理解)
   - [分組策略](#分組策略)
   - [觸發詞設計](#觸發詞設計)
   - [訓練參數](#訓練參數)
3. [ControlNet 訓練](#controlnet-訓練)
   - [Pose Control](#pose-control)
   - [Depth Control](#depth-control)
   - [組合控制](#組合控制)
4. [進階技巧](#進階技巧)
   - [LoRA 組合](#lora-組合)
   - [風格混合](#風格混合)
   - [特效注入](#特效注入)
5. [常見問題](#常見問題)

---

## 特殊場景訓練

### 召喚特效 LoRA

訓練妖怪召喚時的華麗特效（魔法陣、光束、粒子等）。

#### 第一步：檢測召喚場景

```bash
python3 scripts/tools/yokai_summon_scene_detector.py \
    /home/b0979/yokai_input_fast \
    --output-dir /path/to/summon_scenes \
    --extract-mode sample \
    --min-score 60.0
```

**參數說明**：
- `--extract-mode`:
  - `key`: 只提取關鍵幀（最少圖片，最高品質）
  - `sample`: 採樣提取（平衡品質和數量）
  - `all`: 全部提取（最多圖片）
- `--min-score`: 場景評分門檻（0-100）
  - 60-70: 一般召喚場景
  - 70-85: 華麗召喚場景
  - 85+: 超華麗召喚場景

**預期輸出**：
```
summon_scenes/
├── S1.01/
│   ├── scene_001234/       # 場景目錄
│   │   ├── scene_001234_frame_001240.png
│   │   ├── scene_001234_frame_001242.png
│   │   └── ...
│   └── scene_002456/
└── summon_scenes_metadata.json  # 場景評分和元數據
```

#### 第二步：組織特效類型

```bash
python3 scripts/tools/special_effects_organizer.py \
    summon_scenes \
    --output-dir organized_effects \
    --separate-layers
```

**輸出結構**：
```
organized_effects/
├── by_type/
│   ├── summon/              # 召喚類特效（medal召喚、mirapo、portal）
│   ├── attack_soultimate/   # 必殺技特效（含 cut-in、大字）
│   ├── attack_beam/         # 光束攻擊
│   ├── attack_aoe/          # 範圍攻擊
│   ├── magic_circle/        # 魔法陣、封印陣
│   ├── transformation/      # 變身特效（shadowside、godside、fusion）
│   ├── device_watch/        # 妖怪錶光（召喚錶光、錶盤旋轉）
│   ├── yokai_world_mist/    # 妖界紫霧
│   ├── festival_night/      # 祭典特效（燈籠、煙火）
│   ├── pure_effect/         # **純特效（無角色畫面）**
│   └── ambient/             # 環境特效
├── pure_effects/            # 純特效圖層（移除角色）
└── combined/                # 特效+角色組合
```

**重要：純特效檢測**
- 工具會自動檢測「只有特效、沒有角色」的畫面
- 這些畫面會標記為 `pure_effect` 並加上 `is_pure_effect: true` 的 metadata
- **這些畫面不應該用於角色 style/pose 分析，避免汙染訓練資料**
- 純特效畫面適合用於「特效 LoRA」訓練，可以學習純粹的特效風格

#### 第三步：生成 Captions

```bash
python3 scripts/tools/batch_generate_captions_yokai.py \
    organized_effects/by_type/summon \
    --caption-mode effects
```

**Caption 範例**：
```
"yokai summon effect, magic circle, bright light beams, particle effects, energy aura, glowing symbols"
```

#### 第四步：準備訓練資料

```bash
python3 scripts/tools/prepare_yokai_lora_training.py \
    organized_effects/by_type/summon \
    --output-dir training_data/summon_effects \
    --lora-type effects
```

#### 第五步：訓練 LoRA

```bash
accelerate launch train_network.py \
    --config_file training_data/summon_effects/configs/summon_effects_config.toml
```

**建議訓練參數**（特效 LoRA）：
```toml
[model]
pretrained_model_name_or_path = "animefull-latest"

[network]
network_module = "networks.lora"
network_dim = 64              # 特效細節豐富，使用較高維度
network_alpha = 32
network_train_unet_only = true

[training]
max_train_epochs = 25         # 特效需要更多訓練
learning_rate = 1e-4
unet_lr = 1e-4
text_encoder_lr = 5e-5

lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 3   # 多次循環提升細節

# 特效訓練關鍵設定
enable_bucket = true
bucket_no_upscale = true      # 保持原始特效解析度
color_aug = false             # 不要改變特效顏色
flip_aug = false              # 不要翻轉特效方向
```

#### 使用方法

生成召喚特效：
```
Prompt: "jibanyan, yokai summon effect, magic circle, glowing"
LoRA: summon_effects.safetensors (weight: 0.7-0.9)
```

---

### 攻擊特效 LoRA

類似召喚特效，但針對戰鬥場景的攻擊動畫。

#### 訓練流程

```bash
# 1. 從組織好的特效中選擇攻擊類型
mkdir attack_effects_training
cp -r organized_effects/by_type/attack/* attack_effects_training/

# 2. 生成 Captions
python3 scripts/tools/batch_generate_captions_yokai.py \
    attack_effects_training \
    --caption-mode effects

# 3. 準備訓練
python3 scripts/tools/prepare_yokai_lora_training.py \
    attack_effects_training \
    --output-dir training_data/attack_effects \
    --lora-type effects

# 4. 訓練
accelerate launch train_network.py \
    --config_file training_data/attack_effects/configs/attack_effects_config.toml
```

**Caption 範例**：
```
"yokai attack effect, energy blast, shockwave, impact particles, battle scene"
```

---

### 動作序列 (AnimateDiff)

訓練妖怪的動作模式（出場、攻擊、移動等）用於 AnimateDiff motion LoRA。

#### 第一步：提取動作序列

```bash
python3 scripts/tools/action_sequence_extractor.py \
    /home/b0979/yokai_input_fast \
    --output-dir motion_sequences \
    --lengths 16 32 \
    --format animatediff
```

**參數說明**：
- `--lengths`: 序列長度（8/16/32/64 幀）
  - 16: 短動作（出場、轉身）
  - 32: 中等動作（攻擊、移動）
  - 64: 長動作（完整召喚序列）

**輸出結構**：
```
motion_sequences/
├── S1.01_seq001234_len16_entrance/
│   ├── 0000.png
│   ├── 0001.png
│   ├── ...
│   ├── 0015.png
│   └── metadata.json
└── sequences_metadata.json
```

#### 第二步：按動作類型分類

```bash
# 自動分類已完成，查看 metadata.json
cat motion_sequences/sequences_metadata.json | jq '.sequences[] | {name, action_type}'

# 手動選擇特定類型
mkdir entrance_sequences
find motion_sequences -name "*_entrance" -type d -exec cp -r {} entrance_sequences/ \;
```

**動作類型**：
- `entrance`: 出場動畫
- `attack`: 攻擊動作
- `special_effect`: 特效動畫
- `acceleration`: 加速運動
- `transformation`: 變身過程

#### 第三步：AnimateDiff 訓練

**注意**：AnimateDiff motion LoRA 訓練需要特殊的訓練腳本。

```bash
# 使用 AnimateDiff 訓練腳本（需另外安裝）
python train_motion_lora.py \
    --data_dir entrance_sequences \
    --output_dir motion_lora/entrance \
    --sequence_length 16 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_train_steps 5000
```

**訓練要點**：
- 序列長度必須一致（全部 16 或全部 32）
- 批次大小通常為 1（顯存限制）
- 學習率比普通 LoRA 低（1e-4 ~ 5e-5）
- 訓練步數較多（3000-10000 steps）

#### 使用方法

```python
# 在 AnimateDiff pipeline 中使用
from diffusers import AnimateDiffPipeline, MotionAdapter

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "animefull-latest",
    motion_adapter=adapter
)

# 加載 motion LoRA
pipe.load_lora_weights("motion_lora/entrance/entrance_motion.safetensors")

# 生成動畫
output = pipe(
    prompt="jibanyan, entrance animation",
    num_frames=16,
    guidance_scale=7.5
)
```

---

## 風格 LoRA 訓練

### 概念理解

**風格 LoRA vs 角色 LoRA**：

| 類型 | 目標 | 觸發詞 | 訓練資料 |
|------|------|--------|----------|
| 角色 LoRA | 學習特定角色 | "jibanyan" | 單一角色的多張圖片 |
| 風格 LoRA | 學習共同風格 | "cat-type yokai" | 多個相似角色的圖片 |

**為什麼需要風格 LoRA？**

1. **生成變體**：
   - 角色 LoRA: "生成吉胖喵"
   - 風格 LoRA: "生成貓型妖怪風格的新角色"

2. **組合使用**：
   ```
   角色 LoRA (0.8) + 風格 LoRA (0.5)
   = "吉胖喵，但具有更強的貓型妖怪特徵"
   ```

3. **數據效率**：
   - 單一角色資料不足 → 組合多個相似角色 → 學習共同風格

---

### 分組策略

#### 第一步：AI 自動分類

```bash
python3 scripts/tools/yokai_style_classifier.py \
    character_clusters \
    --output-json yokai_taxonomy.json \
    --threshold 0.3 \
    --sample-size 10
```

**互動式審核**：
```
[1/128] cluster_000
AI Classification:
  appearance:
    - animal_cat: 0.85
    - quadruped: 0.72
  attribute:
    - fire: 0.68
  style:
    - cute: 0.91

Action (a/m/s/q): a  # Accept
```

**分類維度**：

1. **Appearance** (外觀)：
   - `animal_cat`: 貓型
   - `animal_dog`: 犬型
   - `animal_bird`: 鳥型
   - `animal_dragon`: 龍型
   - `object_food`: 食物型
   - `object_tool`: 工具型
   - `humanoid`: 人型
   - `ghost`: 幽靈型

2. **Attribute** (屬性)：
   - `fire`: 火
   - `water`: 水
   - `wind`: 風
   - `thunder`: 雷
   - `earth`: 土
   - `ice`: 冰
   - `light`: 光
   - `dark`: 暗

3. **Style** (風格)：
   - `cute`: 可愛
   - `cool`: 酷炫
   - `brave`: 勇敢
   - `scary`: 恐怖
   - `funny`: 搞笑

4. **Body Type** (體型)：
   - `quadruped`: 四足
   - `bipedal`: 二足
   - `flying`: 飛行
   - `floating`: 漂浮

#### 第二步：定義訓練組

創建 `concept_groups.json`：

```json
[
  {
    "name": "cat_type_yokai",
    "category": "appearance",
    "values": ["animal_cat"],
    "comment": "所有貓型妖怪"
  },
  {
    "name": "cute_yokai",
    "category": "style",
    "values": ["cute"],
    "comment": "可愛風格妖怪"
  },
  {
    "name": "fire_water_dual",
    "category": "attribute",
    "values": ["fire", "water"],
    "comment": "火水雙屬性"
  },
  {
    "name": "cute_cat_fire",
    "category": "multi",
    "filters": {
      "appearance": ["animal_cat"],
      "style": ["cute"],
      "attribute": ["fire"]
    },
    "comment": "可愛的火屬性貓型妖怪"
  }
]
```

**分組建議**：

- **單一維度** (推薦初學者)：
  ```json
  {"name": "cat_type", "category": "appearance", "values": ["animal_cat"]}
  ```

- **多標籤單維度**：
  ```json
  {"name": "all_animals", "category": "appearance",
   "values": ["animal_cat", "animal_dog", "animal_bird"]}
  ```

- **多維度組合** (進階)：
  ```json
  {"name": "cute_animals", "category": "multi",
   "filters": {
     "appearance": ["animal_cat", "animal_dog"],
     "style": ["cute"]
   }}
  ```

#### 第三步：準備多概念訓練

```bash
python3 scripts/tools/multi_concept_lora_preparer.py \
    character_clusters \
    --taxonomy yokai_taxonomy.json \
    --output-dir multi_concept_training \
    --groups concept_groups.json \
    --training-type concept
```

**輸出結構**：
```
multi_concept_training/
├── cat_type_yokai/
│   ├── 12_cat_type_yokai/           # Repeat count = 12
│   │   ├── cluster_000_img001.png
│   │   ├── cluster_000_img001.txt   # "yokai, cat-type, char000, ..."
│   │   ├── cluster_005_img023.png
│   │   ├── cluster_005_img023.txt   # "yokai, cat-type, char005, ..."
│   │   └── ...
│   ├── validation/
│   │   └── ...
│   └── configs/
│       └── cat_type_yokai_config.toml
└── multi_concept_metadata.json
```

---

### 觸發詞設計

#### 階層式觸發詞系統

```
Level 1: "yokai"                           → 通用妖怪風格
Level 2: "yokai, cat-type"                 → 貓型妖怪風格
Level 3: "yokai, cat-type, char000"        → 特定貓型妖怪 (cluster_000)
```

**使用範例**：

1. **生成新的貓型妖怪**：
   ```
   Prompt: "yokai, cat-type, orange fur, playful"
   → 生成具有貓型風格的新角色
   ```

2. **生成特定角色**：
   ```
   Prompt: "yokai, cat-type, char000, sitting"
   → 生成 cluster_000 的妖怪（坐姿）
   ```

3. **組合多個層級**：
   ```
   Prompt: "yokai, cat-type, fire attribute, cute style"
   → 可愛的火屬性貓型妖怪
   ```

#### 觸發詞配置

在 `multi_concept_lora_preparer.py` 中自動生成：

```python
# 階層式
trigger_words = "yokai, cat-type, char000"

# 扁平式（可選）
trigger_words = "cat_type_yokai_char000"
```

**選擇建議**：
- **階層式**: 更靈活，可以只用部分觸發詞
- **扁平式**: 更簡單，不會混淆

---

### 訓練參數

#### 根據組大小自動調整

```python
# 3 個角色以下 - 學習個體 + 共同風格
if num_characters <= 3:
    network_dim = 64      # 高維度保留個體特徵
    network_alpha = 32
    unet_lr = 1e-4
    max_train_epochs = 20

# 4-10 個角色 - 平衡個體和風格
elif num_characters <= 10:
    network_dim = 48
    network_alpha = 24
    unet_lr = 1.2e-4
    max_train_epochs = 18

# 10+ 個角色 - 聚焦共同風格
else:
    network_dim = 32      # 低維度強制學習共同特徵
    network_alpha = 16
    unet_lr = 1.5e-4
    max_train_epochs = 15
```

#### 手動調整技巧

**如果風格不夠明顯**：
```toml
network_dim = 32          # 降低維度
network_alpha = 16
unet_lr = 1.5e-4          # 提高學習率
max_train_epochs = 20     # 增加訓練輪數
```

**如果個體特徵丟失**：
```toml
network_dim = 64          # 提高維度
network_alpha = 32
unet_lr = 8e-5            # 降低學習率
max_train_epochs = 15
```

#### 訓練配置範例

```toml
[model]
pretrained_model_name_or_path = "animefull-latest"

[network]
network_module = "networks.lora"
network_dim = 48                    # 自動計算
network_alpha = 24
network_train_unet_only = true
network_dropout = 0.1               # 防止過擬合

[training]
max_train_epochs = 18
learning_rate = 1.2e-4
unet_lr = 1.2e-4
text_encoder_lr = 6e-5

lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 2

optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.1"]  # 風格訓練使用較高 weight decay

# 多概念訓練關鍵設定
shuffle_caption = true              # 打亂 caption 提升泛化
keep_tokens = 2                     # 保持 "yokai, cat-type" 在前
caption_dropout_rate = 0.05         # 輕微 dropout 提升泛化

# 數據增強
enable_bucket = true
color_aug = false                   # 保持風格顏色一致性
flip_aug = true                     # 可以翻轉
```

---

## ControlNet 訓練

### Pose Control

使用 OpenPose 控制妖怪姿態。

#### 第一步：生成 OpenPose 資料集

```bash
python3 scripts/tools/controlnet_complete_pipeline.py \
    character_clusters/cluster_000 \
    --output-dir controlnet_pose_data \
    --control-types openpose
```

**輸出結構**：
```
controlnet_pose_data/
├── source/              # 原圖
│   └── img001.png
├── openpose/            # Pose 骨架圖
│   └── img001.png
└── captions/
    └── img001.txt
```

#### 第二步：ControlNet 訓練

使用 `sd-scripts` 的 ControlNet 訓練腳本：

```bash
accelerate launch train_controlnet.py \
    --pretrained_model_name_or_path="animefull-latest" \
    --train_data_dir="controlnet_pose_data" \
    --conditioning_image_column="openpose" \
    --image_column="source" \
    --caption_column="captions" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=4 \
    --max_train_steps=50000 \
    --output_dir="controlnet_pose_yokai"
```

**訓練要點**：
- ControlNet 訓練需要更多資料（建議 1000+ 組）
- 學習率較低（1e-5）
- 訓練時間長（50k steps）
- 建議使用多 GPU

#### 使用方法

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained("controlnet_pose_yokai")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "animefull-latest",
    controlnet=controlnet
)

# 使用 OpenPose 圖控制生成
image = pipe(
    prompt="jibanyan, cute pose",
    image=openpose_image,
    num_inference_steps=20
).images[0]
```

---

### Depth Control

使用深度圖控制場景深度。

#### 動畫風格深度生成

```bash
python3 scripts/tools/controlnet_complete_pipeline.py \
    character_clusters/cluster_000 \
    --output-dir controlnet_depth_data \
    --control-types depth \
    --background-dir layered_frames/background
```

**深度圖特點**：
- 角色層：深度 0.7-1.0（最近）
- 背景層：深度 0.0-0.5（最遠）
- 邊緣保持（不模糊動畫線條）

#### 訓練方法

同 Pose Control，將 `--conditioning_image_column` 改為 `depth`。

---

### 組合控制

同時使用多個 ControlNet。

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, MultiControlNetModel

# 載入多個 ControlNet
controlnet_pose = ControlNetModel.from_pretrained("controlnet_pose_yokai")
controlnet_depth = ControlNetModel.from_pretrained("controlnet_depth_yokai")

controlnets = MultiControlNetModel([controlnet_pose, controlnet_depth])

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "animefull-latest",
    controlnet=controlnets
)

# 使用多個控制圖
image = pipe(
    prompt="jibanyan, outdoor scene",
    image=[openpose_image, depth_image],
    controlnet_conditioning_scale=[0.8, 0.5],  # Pose 權重高，depth 權重低
    num_inference_steps=20
).images[0]
```

---

## 進階技巧

### LoRA 組合

組合多個 LoRA 實現複雜效果。

#### 範例 1: 角色 + 特效

```python
# 載入基礎模型
pipe = StableDiffusionPipeline.from_pretrained("animefull-latest")

# 載入角色 LoRA (吉胖喵)
pipe.load_lora_weights("lora/jibanyan.safetensors", adapter_name="character")

# 載入特效 LoRA (召喚特效)
pipe.load_lora_weights("lora/summon_effects.safetensors", adapter_name="effects")

# 設定權重
pipe.set_adapters(["character", "effects"], adapter_weights=[0.8, 0.7])

# 生成
image = pipe(
    prompt="jibanyan, yokai summon effect, magic circle, glowing",
    num_inference_steps=30
).images[0]
```

**權重建議**：
- 角色 LoRA: 0.7-0.9（主體）
- 特效 LoRA: 0.5-0.8（輔助）
- 風格 LoRA: 0.4-0.6（微調）

#### 範例 2: 角色 + 風格 + 特效

```python
pipe.load_lora_weights("lora/jibanyan.safetensors", adapter_name="char")
pipe.load_lora_weights("lora/cat_type_yokai.safetensors", adapter_name="style")
pipe.load_lora_weights("lora/fire_effects.safetensors", adapter_name="fx")

pipe.set_adapters(["char", "style", "fx"], adapter_weights=[0.8, 0.5, 0.6])

image = pipe(
    prompt="jibanyan, cat-type yokai style, fire attribute, attack effect",
    num_inference_steps=30
).images[0]
```

**效果**：
- `char (0.8)`: 吉胖喵的基本外觀
- `style (0.5)`: 強化貓型妖怪特徵
- `fx (0.6)`: 添加火焰攻擊特效

---

### 風格混合

混合不同風格 LoRA 創造新風格。

#### 貓型 + 犬型 = 混合型

```python
pipe.load_lora_weights("lora/cat_type.safetensors", adapter_name="cat")
pipe.load_lora_weights("lora/dog_type.safetensors", adapter_name="dog")

# 各 50% 權重
pipe.set_adapters(["cat", "dog"], adapter_weights=[0.5, 0.5])

image = pipe(
    prompt="yokai, animal hybrid, cute",
    num_inference_steps=30
).images[0]
```

#### 可愛 + 酷炫 = 平衡風格

```python
pipe.load_lora_weights("lora/cute_yokai.safetensors", adapter_name="cute")
pipe.load_lora_weights("lora/cool_yokai.safetensors", adapter_name="cool")

pipe.set_adapters(["cute", "cool"], adapter_weights=[0.6, 0.4])

image = pipe(
    prompt="yokai, balanced style, adorable yet stylish"
).images[0]
```

---

### 特效注入

在角色生成後期注入特效。

#### 兩階段生成

```python
# 階段 1: 生成角色（無特效）
pipe.set_adapters(["character"], adapter_weights=[0.8])
latents = pipe(
    prompt="jibanyan, standing pose",
    num_inference_steps=30,
    output_type="latent"
).images

# 階段 2: 注入特效（後期步驟）
pipe.set_adapters(["character", "effects"], adapter_weights=[0.6, 0.9])
image = pipe(
    prompt="jibanyan, summon effect, magic circle, glowing",
    latents=latents,
    num_inference_steps=15,  # 額外 15 步
    strength=0.5  # 保留 50% 原圖
).images[0]
```

**優點**：
- 角色細節保留更好
- 特效更自然地融合

#### img2img + 特效

```python
# 使用已生成的角色圖
base_image = Image.open("jibanyan_base.png")

# 只載入特效 LoRA
pipe.load_lora_weights("lora/summon_effects.safetensors")

# img2img 添加特效
image = pipe(
    prompt="yokai summon effect, magic circle, particle effects",
    image=base_image,
    strength=0.4,  # 保留 60% 原圖
    num_inference_steps=20
).images[0]
```

---

## 常見問題

### Q1: 風格 LoRA 訓練後，角色特徵丟失了？

**原因**：維度太低或學習率太高。

**解決方案**：
```toml
network_dim = 64      # 提高維度 (原本 32)
network_alpha = 32
unet_lr = 8e-5        # 降低學習率 (原本 1.5e-4)
max_train_epochs = 15 # 減少訓練輪數
```

### Q2: 多概念訓練時，觸發詞不起作用？

**檢查**：
1. Caption 中是否包含階層觸發詞？
   ```
   ✓ "yokai, cat-type, char000, sitting"
   ✗ "char000, sitting"
   ```

2. `keep_tokens` 設定：
   ```toml
   keep_tokens = 2  # 保持 "yokai, cat-type" 在前
   ```

3. 訓練時是否使用 `shuffle_caption`：
   ```toml
   shuffle_caption = true
   ```

### Q3: 特效 LoRA 生成的特效太弱？

**解決方案**：

1. **提高 LoRA 權重**：
   ```python
   pipe.set_adapters(["effects"], adapter_weights=[0.9])  # 從 0.7 提高到 0.9
   ```

2. **增強 Prompt**：
   ```
   "yokai summon effect, strong magic circle, very bright light beams,
    intense particle effects, glowing energy, spectacular"
   ```

3. **調整訓練參數**：
   ```toml
   network_dim = 128     # 提高維度捕捉更多特效細節
   max_train_epochs = 30 # 更多訓練
   ```

### Q4: ControlNet Pose 生成時姿態不準確？

**解決方案**：

1. **檢查 OpenPose 圖品質**：
   - 骨架是否完整？
   - 關節位置是否正確？

2. **提高 ControlNet 權重**：
   ```python
   controlnet_conditioning_scale=1.0  # 從 0.8 提高到 1.0
   ```

3. **使用更好的 OpenPose 模型**：
   ```python
   from controlnet_aux import OpenposeDetector
   openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
   ```

4. **訓練妖怪特定的 ControlNet**（非人型）：
   - 收集更多非人型姿態資料
   - 使用動物骨架模型

### Q5: 組合多個 LoRA 時，效果互相衝突？

**原因**：權重總和過高或 LoRA 之間不兼容。

**解決方案**：

1. **降低總權重**：
   ```python
   # ✗ 總和 = 2.4
   pipe.set_adapters(["char", "style", "fx"], adapter_weights=[0.9, 0.8, 0.7])

   # ✓ 總和 = 1.7
   pipe.set_adapters(["char", "style", "fx"], adapter_weights=[0.7, 0.5, 0.5])
   ```

2. **優先順序**：
   - 主要 LoRA (角色): 0.7-0.9
   - 次要 LoRA (風格/特效): 0.4-0.6

3. **測試兼容性**：
   ```python
   # 逐個添加測試
   # 1. 只有角色
   # 2. 角色 + 風格
   # 3. 角色 + 風格 + 特效
   ```

### Q6: AnimateDiff motion LoRA 訓練失敗？

**常見問題**：

1. **序列長度不一致**：
   ```bash
   # 確保所有序列長度相同
   find motion_sequences -name "*.png" | wc -l  # 應該是 N * 16
   ```

2. **顯存不足**：
   ```bash
   # 減少批次大小
   --batch_size=1

   # 使用梯度檢查點
   --gradient_checkpointing
   ```

3. **學習率太高**：
   ```bash
   # AnimateDiff 學習率應該更低
   --learning_rate=5e-5  # 而非 1e-4
   ```

### Q7: ControlNet 訓練資料不足？

**解決方案**：

1. **數據增強**：
   ```python
   # 翻轉
   # 旋轉（小角度）
   # 縮放
   ```

2. **組合多個角色的資料**：
   ```bash
   # 合併所有 cluster 的 ControlNet 資料
   mkdir combined_controlnet_data
   cp -r controlnet_datasets/*/source/* combined_controlnet_data/source/
   cp -r controlnet_datasets/*/openpose/* combined_controlnet_data/openpose/
   ```

3. **使用預訓練 ControlNet 微調**：
   ```bash
   # 從官方 ControlNet 開始
   --pretrained_controlnet="lllyasviel/control_v11p_sd15_openpose"
   # 只需 5k-10k steps 微調
   ```

---

## 總結

### 訓練流程速查

1. **特效 LoRA**:
   ```
   summon_detector → effects_organizer → caption → prepare → train
   ```

2. **風格 LoRA**:
   ```
   style_classifier → groups.json → multi_concept_preparer → train
   ```

3. **Motion LoRA**:
   ```
   action_extractor → AnimateDiff train
   ```

4. **ControlNet**:
   ```
   controlnet_pipeline → ControlNet train
   ```

### 推薦訓練順序

1. ✅ 角色 LoRA（基礎）
2. ✅ 風格 LoRA（擴展）
3. ✅ 特效 LoRA（增強）
4. 🔧 ControlNet（精確控制）
5. 🔧 Motion LoRA（動畫生成）

### 參考資源

- **基礎訓練**: `USAGE_GUIDE.md`
- **工具快速參考**: `TOOLS_QUICK_REFERENCE.md`
- **進階工具**: `ADVANCED_FEATURES_QUICK_START.md`
- **kohya_ss 文檔**: https://github.com/kohya-ss/sd-scripts

---

**最後更新**: 2025-10-30
**版本**: v1.0
**適用於**: Yokai Watch LoRA 訓練項目
