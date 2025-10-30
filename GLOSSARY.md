# Terminology Glossary (術語對照表)

English-Chinese terminology reference for multi-anime LoRA training.

英中術語對照，用於多動畫 LoRA 訓練專案。

---

## Core Concepts (核心概念)

| English | Chinese | Description |
|---------|---------|-------------|
| **Segmentation** | 分割 / 分層 | Separating character and background layers |
| **Clustering** | 聚類 | Grouping similar characters together |
| **LoRA** | LoRA (Low-Rank Adaptation) | Efficient fine-tuning method for large models |
| **Caption** | 標註 / 描述文字 | Text description of image content |
| **Training Data** | 訓練資料 | Images and metadata used for model training |
| **Dataset** | 資料集 | Organized collection of training data |
| **Pipeline** | 管道 / 流程 | Automated processing workflow |
| **Batch Processing** | 批次處理 | Processing multiple files at once |
| **Taxonomy** | 分類體系 | Hierarchical classification system |
| **Schema** | 架構 / 模式 | Structured data format |

## Image Processing (影像處理)

| English | Chinese | Description |
|---------|---------|-------------|
| **Character Layer** | 角色層 | Segmented character without background |
| **Background Layer** | 背景層 | Scene without characters |
| **Effect Layer** | 特效層 | Visual effects (sparkles, aura, etc.) |
| **Pure Effect** | 純特效 | Effects without characters |
| **Frame** | 幀 / 影格 | Single image from video |
| **Keyframe** | 關鍵幀 | Important frame in a sequence |
| **Mask** | 遮罩 | Binary image defining regions |
| **Alpha Channel** | Alpha 通道 | Transparency information |
| **RGBA** | RGBA | Red, Green, Blue, Alpha color model |
| **Edge Detection** | 邊緣檢測 | Finding boundaries in images |
| **Depth Map** | 深度圖 | Distance information for each pixel |

## Models & AI (模型與AI)

| English | Chinese | Description |
|---------|---------|-------------|
| **U2-Net** | U2-Net | Segmentation model |
| **YOLO** | YOLO | Object detection model |
| **CLIP** | CLIP | Vision-language model |
| **BLIP-2** | BLIP-2 | Image captioning model |
| **DINOv2** | DINOv2 | Feature extraction model |
| **HDBSCAN** | HDBSCAN | Density-based clustering algorithm |
| **OpenPose** | OpenPose | Pose estimation model |
| **ControlNet** | ControlNet | Conditional image generation control |
| **Stable Diffusion** | Stable Diffusion (穩定擴散) | Text-to-image generation model |
| **SDXL** | SDXL (Stable Diffusion XL) | Larger Stable Diffusion model |

## Training (訓練)

| English | Chinese | Description |
|---------|---------|-------------|
| **Fine-tuning** | 微調 | Adapting pre-trained model to specific data |
| **Epoch** | 訓練輪次 | Complete pass through training dataset |
| **Batch Size** | 批次大小 | Number of samples processed together |
| **Learning Rate** | 學習率 | Step size in gradient descent |
| **Network Dimension** | 網路維度 | LoRA model capacity |
| **Network Alpha** | 網路 Alpha | LoRA regularization parameter |
| **Repeat** | 重複次數 | How many times to use each image |
| **Checkpoint** | 檢查點 | Saved model state |
| **Validation** | 驗證 | Testing model on held-out data |
| **Overfitting** | 過擬合 | Model memorizes training data |

## Yokai Watch Specific (妖怪手錶專用)

| English | Chinese | Description |
|---------|---------|-------------|
| **Summon Scene** | 召喚場景 | Yokai summoning animation |
| **Soultimate** | 必殺技 | Special finishing move |
| **Inspirit** | 附身 / 妖氣 | Yokai possession ability |
| **Medal** | 妖怪獎章 | Token used to summon yokai |
| **Yo-kai Watch** | 妖怪手錶 | Device for seeing/summoning yokai |
| **Shadowside** | 影之側 | Dark transformation form |
| **Godside** | 神之側 | Divine transformation form |
| **Enma Palace** | 閻魔宮 | King Enma's palace |
| **Gera Gera Land** | Gera Gera 樂園 | Yokai amusement park |
| **Oni Time** | 鬼時間 | Dangerous time period |
| **Terror Time** | 恐怖時間 | Another name for Oni Time |
| **Mirapo** | Mirapo傳送 | Teleportation system |

## Scene Types (場景類型)

| English | Chinese | Description |
|---------|---------|-------------|
| **Realm** | 世界 / 界域 | Major world division |
| **Location** | 地點 / 位置 | Specific place within realm |
| **Environment** | 環境類型 | Indoor/outdoor/fantasy classification |
| **Activity** | 活動 / 情境 | What's happening in scene |
| **Audio Environment** | 音訊環境 | Ambient sound characteristics |
| **Human Town** | 人類城鎮 | Normal human world |
| **Yo-kai World** | 妖怪世界 | Spirit world |
| **Special Dimension** | 特殊空間 | Unique supernatural spaces |
| **Battle Scene** | 戰鬥場景 | Combat sequence |
| **Daily Life** | 日常生活 | Normal activities |

## Body Types (身體類型)

| English | Chinese | Description |
|---------|---------|-------------|
| **Humanoid** | 人型 | Human-like form |
| **Quadruped** | 四足獸型 | Four-legged animal |
| **Bipedal** | 雙足 | Two-legged |
| **Flying** | 飛行型 | Can fly independently |
| **Floating** | 漂浮型 | Hovers without wings |
| **Cloud Rider** | 乘雲型 | Rides on clouds |
| **Multi-limb** | 多肢型 | More than 4 limbs |
| **Tentacle** | 觸手型 | Has tentacles |
| **Serpentine** | 蛇型 | Snake-like long body |
| **Object Form** | 物件型 | Inanimate object yokai |
| **Robot/Mecha** | 機器人/機甲型 | Mechanical form |

## Effects (特效)

| English | Chinese | Description |
|---------|---------|-------------|
| **Summon Effect** | 召喚特效 | Effects during summoning |
| **Attack Effect** | 攻擊特效 | Combat visual effects |
| **Beam** | 光束 | Straight light beam attack |
| **AOE (Area of Effect)** | 範圍技 | Area-wide attack |
| **Magic Circle** | 魔法陣 | Glowing ritual circle |
| **Seal/Barrier** | 結界 / 護盾 | Protective energy field |
| **Transformation** | 變身 / 進化 | Form change effect |
| **Fusion** | 合體 / 融合 | Combining multiple entities |
| **Aura** | 氣場 / 光環 | Energy emanation |
| **Particle Effect** | 粒子特效 | Sparkles, dust, etc. |
| **Cut-in** | Cut-in 特寫 | Close-up dramatic shot |
| **Kanji Text** | 大字特效 | Large text overlay |
| **Speedlines** | 速度線 | Motion indicator lines |

## File Organization (檔案組織)

| English | Chinese | Description |
|---------|---------|-------------|
| **Core** | 核心 | Shared fundamental code |
| **Generic** | 通用 | Anime-agnostic tools |
| **Anime-Specific** | 動畫專屬 | Designed for one anime series |
| **Tools** | 工具 | Reusable utility scripts |
| **Pipelines** | 管道 / 流程腳本 | End-to-end processing scripts |
| **Batch** | 批次腳本 | Mass processing automation |
| **Evaluation** | 評估 | Quality assessment tools |
| **Tests** | 測試 | Validation and testing scripts |

## Technical Terms (技術術語)

| English | Chinese | Description |
|---------|---------|-------------|
| **Repository** | 儲存庫 / 程式碼庫 | Code storage location |
| **Directory** | 目錄 / 資料夾 | Folder containing files |
| **Import** | 匯入 | Loading code from another module |
| **Dependency** | 依賴項 | Required external library |
| **Environment Variable** | 環境變數 | System configuration value |
| **GPU** | GPU (圖形處理器) | Graphics processing unit |
| **CUDA** | CUDA | NVIDIA GPU programming framework |
| **CLI** | 命令列介面 | Command-line interface |
| **API** | API (應用程式介面) | Application programming interface |
| **JSON** | JSON | JavaScript Object Notation data format |
| **TOML** | TOML | Configuration file format |
| **Metadata** | 元數據 / 詮釋資料 | Data about data |

---

## Usage Notes (使用說明)

### In Documentation (文檔中)
Use English terms first, with Chinese in parentheses for key concepts:
```markdown
The **segmentation (分割)** module separates character and background layers.
```

### In Code Comments (程式碼註解中)
```python
# Character clustering (角色聚類) using HDBSCAN
def cluster_characters(features, min_cluster_size=25):
    """
    Cluster character features into groups.
    使用 HDBSCAN 將角色特徵分組。
    """
    pass
```

### In File Names (檔案名稱中)
Use English only, no Chinese characters:
```
✅ yokai_summon_detector.py
✅ character_clustering.py
❌ 妖怪召喚檢測器.py
```

---

**Last Updated**: 2025-10-30
**Version**: 2.0 (Project Reorganization)
