# 使用指南

本指南說明如何使用自動化 pipeline 訓練閃電十一人角色的 LoRA 模型。

## 快速開始

### 前置步驟：安裝環境

```bash
# 1. 安裝 PyTorch (RTX 5080 專用)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 2. 安裝其他依賴
pip install -r requirements.txt

# 3. 驗證安裝
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 使用場景

### 場景 A：使用現有的 Gold Standard 圖片（円堂守）

你已經手動挑選了 67 張円堂守的高品質圖片。現在要準備這些圖片用於訓練：

```bash
# 一鍵準備訓練數據
python scripts/tools/prepare_training_data.py endou_mamoru --style detailed --repeat 10
```

這個腳本會：
1. ✅ 使用 WD14 Tagger v3 為每張圖片生成詳細標註
2. ✅ 組織成 kohya_ss 訓練格式 (`data/characters/endou_mamoru/training_ready/10_endou_mamoru/`)
3. ✅ 生成配對的 `.txt` caption 文件

**預期輸出：**
- 67 張圖片 + 67 個 caption 文件
- 每個 epoch 會重複 10 次（有效訓練樣本 = 670）

---

### 場景 B：從影片自動收集角色圖片（其他角色）

當你要訓練其他角色但沒有足夠圖片時，使用完整 pipeline 從影片自動提取：

#### 步驟 1：準備影片和配置

1. 將動畫影片放到 `data/raw_videos/`
2. 準備少量（10-20張）該角色的 gold_standard 參考圖
3. 編輯角色配置文件（參考 `config/characters/endou_mamoru.yaml`）

#### 步驟 2：執行完整 Pipeline

```bash
python scripts/pipeline/pipeline_orchestrator.py gouenji_shuuya \
  --videos data/raw_videos/S01E01.mkv data/raw_videos/S01E02.mkv
```

這會自動執行：
1. 📹 **VideoProcessor**: 從影片提取場景關鍵幀
2. 🧹 **ImageCleaner**: 去模糊、去重、品質過濾
3. 🎯 **CharacterFilter**: 兩階段角色識別（WD14 + CLIP）
4. 📝 **AutoCaptioner**: 生成訓練標註
5. 📦 **PrepareTraining**: 組織 kohya_ss 格式

**預期輸出：**
- 自動篩選出該角色的圖片
- 已生成標註並準備訓練

---

## 進階用法

### 單獨使用各模組

#### 1. 只生成 Caption

```bash
python scripts/tools/caption_gold_standard.py endou_mamoru --style detailed
```

#### 2. 只做角色過濾

```bash
python scripts/pipeline/character_filter.py \
  data/candidates/ \
  --gold-standard data/characters/endou_mamoru/gold_standard/v1.0/images \
  --output data/filtered/ \
  --clip-threshold 0.75
```

#### 3. 只做影片處理

```bash
python scripts/pipeline/video_processor.py \
  data/raw_videos/S01E01.mkv \
  --output data/extracted_frames/
```

#### 4. 只做去重清理

```bash
python scripts/pipeline/image_cleaner.py \
  data/raw_images/ \
  --output data/cleaned/
```

---

## 訓練配置

### Kohya_ss 訓練參數

準備好訓練數據後，使用 kohya_ss 進行訓練：

```bash
# 假設你已安裝 kohya_ss 到 models/sd-scripts/
cd models/sd-scripts/

python train_network.py \
  --pretrained_model_name_or_path="models/base_models/AnythingV5.safetensors" \
  --train_data_dir="../../data/characters/endou_mamoru/training_ready/" \
  --output_dir="../../models/loras/endou_mamoru/v1/" \
  --output_name="endou_mamoru_v1" \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine_with_restarts" \
  --train_batch_size=4 \
  --max_train_epochs=10 \
  --save_every_n_epochs=2 \
  --mixed_precision="fp16" \
  --optimizer_type="AdamW8bit" \
  --xformers \
  --cache_latents
```

**推薦訓練參數（67 張圖片，repeat=10）：**
- Epochs: 10-15
- Batch size: 4
- Network dim: 32
- Learning rate: 1e-4
- 總訓練步數 ≈ (67 × 10 ÷ 4) × 10 = ~1675 steps

---
## 開始訓練

### 方法 1：使用便利腳本（推薦）

```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora
./train_endou_mamoru.sh
```

腳本會：
1. 檢查 conda 環境
2. 檢查 GPU 狀態
3. 顯示配置摘要
4. 確認後啟動訓練

### 方法 2：直接命令

```bash
cd /mnt/c/AI_LLM_projects/sd-scripts

accelerate launch --num_cpu_threads_per_process=8 \
  train_network.py \
  --config_file=/mnt/c/AI_LLM_projects/inazuma-eleven-lora/train_endou_mamoru.toml
```

### 訓練日誌
```
/mnt/c/AI_LLM_projects/ai_warehouse/outputs/inazuma-eleven/logs/endou_mamoru/
```

---

## 監控訓練

### 使用 TensorBoard

```bash
tensorboard --logdir=/mnt/c/AI_LLM_projects/ai_warehouse/outputs/inazuma-eleven/logs
```

在瀏覽器中打開 `http://localhost:6006` 查看：
- Loss curves
- Learning rate schedule
- Training metrics

---

## 訓練後測試

### 1. 載入 LoRA 並生成測試圖片

建立測試腳本 `test_lora.py`:

```python
from diffusers import StableDiffusionPipeline
import torch

# 載入 base model
pipe = StableDiffusionPipeline.from_single_file(
    "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors",
    torch_dtype=torch.float16
)

# 載入 LoRA
pipe.load_lora_weights(
    "/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven/endou_mamoru_v1.safetensors"
)

pipe.to("cuda")

# 測試提示詞
test_prompts = [
    "endou_mamoru, smiling, looking at viewer, portrait",
    "endou_mamoru, serious expression, goalkeeper pose",
    "endou_mamoru, soccer field background, action pose",
    "endou_mamoru, school uniform, classroom",
]

for i, prompt in enumerate(test_prompts):
    image = pipe(
        prompt,
        num_inference_steps=28,
        guidance_scale=7.0,
        height=512,
        width=512
    ).images[0]

    image.save(f"test_endou_v1_{i+1}.png")
    print(f"✓ Generated: test_endou_v1_{i+1}.png")
```


## 專案結構

```
data/
├── characters/
│   └── endou_mamoru/
│       ├── gold_standard/v1.0/images/     # 手動挑選的高品質圖
│       ├── auto_collected/                 # Pipeline 自動收集的圖
│       └── training_ready/10_endou_mamoru/ # 訓練就緒數據
├── raw_videos/                             # 原始動畫影片
└── cache/                                  # CLIP 快取

models/
├── base_models/                            # SD 基礎模型
├── loras/                                  # 訓練完成的 LoRA
└── sd-scripts/                             # kohya_ss 訓練腳本

config/
├── global_config.yaml                      # 全域配置
└── characters/
    ├── endou_mamoru.yaml                   # 角色配置
    ├── gouenji_shuuya.yaml
    └── ...

scripts/
├── pipeline/                               # 核心模組
│   ├── video_processor.py
│   ├── image_cleaner.py
│   ├── character_filter.py
│   ├── auto_captioner.py
│   └── pipeline_orchestrator.py
├── tools/                                  # 工具腳本
│   ├── prepare_training_data.py
│   └── caption_gold_standard.py
└── utils/                                  # 工具函數
```

---

## 常見問題

### Q1: CUDA out of memory

**方案：**
- 降低 batch_size（在 CharacterFilter 中調整）
- 使用較小的 CLIP 模型（ViT-B/32 而非 ViT-L/14）
- 在配置中啟用 `enable_xformers: true`

### Q2: WD14 Tagger 推理緩慢

**方案：**
- 確認安裝的是 `onnxruntime-gpu` 而非 `onnxruntime`
- 檢查 CUDA 是否可用

### Q3: Character Filter 誤判率高

**方案：**
- 增加 gold_standard 參考圖數量（推薦 20+ 張）
- 調整 `clip_threshold`（降低閾值會保留更多圖片）
- 檢查 `required_tags` 和 `forbidden_tags` 是否過於嚴格

### Q4: Caption 品質不佳

**方案：**
- 調整 `general_threshold`（提高閾值 = 更精確但更少標籤）
- 使用 `detailed` style 而非 `minimal`
- 手動編輯生成的 `.txt` 文件

---

## 如果需要調整

### 問題 1: Loss 下降太慢
```toml
# 增加 learning rate
learning_rate = 2e-4  # 從 1e-4 增加
```

### 問題 2: 過擬合 (Overfitting)
```toml
# 減少 epochs
max_train_epochs = 6  # 從 10 減少

# 或增加 regularization
network_alpha = 8  # 從 16 減少
```

### 問題 3: CUDA OOM (記憶體不足)
```toml
# 減少 batch size
batch_size = 2  # 從 4 減少

# 增加 gradient accumulation
gradient_accumulation_steps = 2
```
---

## BLIP-2 升級 (可選)

當 BLIP-2 模型下載完成後，可以生成更詳細的標註並重新訓練 v2:

```bash
# 1. 測試 BLIP-2
conda run -n blip2-env python3 /tmp/test_blip2.py

# 2. 生成 BLIP-2 標註
conda run -n blip2-env python3 scripts/tools/generate_captions_blip2.py endou_mamoru

# 3. 重新準備訓練數據
python3 scripts/tools/prepare_training_data.py endou_mamoru --style detailed --repeat 10

# 4. 修改配置文件
# 將 output_name 改為 "endou_mamoru_v2"

# 5. 重新訓練
./train_endou_mamoru.sh
```
## 故障排除

### 訓練無法啟動

**檢查環境**:
```bash
conda activate env-ai
python3 -c "import library; print('✓ sd-scripts OK')"
nvidia-smi
```

**檢查路徑**:
```bash
# 確認所有路徑存在
ls /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/characters/endou_mamoru/training_ready/10_endou_mamoru/
ls /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors
```

### 訓練中斷

```bash
# 檢查是否有已保存的 checkpoint
ls -lh /mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven/

# 如果有 checkpoint，可以修改配置繼續訓練：
# 在 train_endou_mamoru.toml 中添加:
# resume = "/path/to/endou_mamoru_v1-000004.safetensors"
```

如遇問題，請檢查：
1. `outputs/logs/` 中的日誌文件
2. 各階段輸出的 `*_report.json` 文件
3. GPU 記憶體使用情況
