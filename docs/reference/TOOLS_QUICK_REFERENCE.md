# Yokai Watch LoRA Tools - Quick Reference

Quick reference for all post-processing tools.

---

## 📦 Complete Pipeline (Automated)

### yokai_lora_complete_pipeline.sh

**Purpose**: Automates entire workflow from episodes to training data

**Usage**:
```bash
./scripts/batch/yokai_lora_complete_pipeline.sh
```

**Environment Variables**:
```bash
INPUT_DIR=/path/to/episodes \
OUTPUT_BASE=/path/to/output \
MIN_CLUSTER_SIZE=15 \
AUGMENT_SMALL_CLUSTERS=true \
PREPARE_BACKGROUNDS=false \
SKIP_STAGES="1,2" \
./scripts/batch/yokai_lora_complete_pipeline.sh
```

**Stages**:
1. Segmentation (U2Net)
2. Clustering (HDBSCAN + CLIP)
3. Cluster analysis
4. Data augmentation
5. Caption generation (BLIP2)
6. Training data preparation
7. Validation
8. Background preparation (optional)

**Duration**: 24-30 hours for 214 episodes

---

## 🔍 Analysis Tools

### 1. analyze_yokai_clusters.py

**Purpose**: Evaluate cluster quality and get training recommendations

**Basic Usage**:
```bash
python3 scripts/tools/analyze_yokai_clusters.py \
    /path/to/character_clusters \
    --output-json analysis.json \
    --output-html analysis.html
```

**Advanced Options**:
```bash
python3 scripts/tools/analyze_yokai_clusters.py \
    /path/to/clusters \
    --output-json analysis.json \
    --output-html analysis.html \
    --min-quality 0.6 \
    --min-diversity 0.3
```

**Output**:
- JSON with cluster metrics
- HTML visual report (open in browser)
- Tier rankings (S/A/B/C/D)
- Training recommendations

**Metrics**:
- Quality score (0-1): Sharpness, brightness, contrast
- Diversity score (0-1): Pose/expression variety
- Character type: Human, Yokai, Unknown
- Tier: S (best) to D (skip)

---

## 🎨 Data Augmentation

### 2. augment_small_clusters.py

**Purpose**: Expand small sample clusters while preserving character identity

**Basic Usage**:
```bash
python3 scripts/tools/augment_small_clusters.py \
    /path/to/clusters \
    --output-dir /path/to/augmented_clusters
```

**Common Configurations**:

**Standard (5-30 images, medium intensity)**:
```bash
python3 scripts/tools/augment_small_clusters.py \
    /path/to/clusters \
    --output-dir /path/to/augmented \
    --max-original 30 \
    --min-original 5 \
    --aug-intensity medium
```

**Aggressive (very small clusters, heavy augmentation)**:
```bash
python3 scripts/tools/augment_small_clusters.py \
    /path/to/clusters \
    --output-dir /path/to/augmented \
    --max-original 20 \
    --min-original 5 \
    --aug-intensity heavy \
    --target-multiplier 8.0
```

**Light (near threshold, minimal changes)**:
```bash
python3 scripts/tools/augment_small_clusters.py \
    /path/to/clusters \
    --output-dir /path/to/augmented \
    --max-original 40 \
    --min-original 20 \
    --aug-intensity light
```

**Parameters**:
- `--max-original`: Only augment clusters with ≤ this many images
- `--min-original`: Skip clusters with fewer images
- `--aug-intensity`: `light`, `medium`, `heavy`
- `--target-multiplier`: Override auto multiplier (e.g., 4.0 for 4x)
- `--seed`: Random seed for reproducibility

**Auto Multipliers**:
- 5-10 images → 6-8x
- 11-15 images → 4-5x
- 16-20 images → 3-4x
- 21-30 images → 1.5-2x

**Techniques Applied**:
- Horizontal flip
- Rotation (-5° to +5°)
- Brightness (0.8-1.2x)
- Contrast (0.9-1.1x)
- Hue shift (±10°)
- Crop variation (95-100%)
- Slight blur (optional)

---

## 💬 Caption Generation

### 3. batch_generate_captions_yokai.py

**Purpose**: Generate training captions using BLIP2

**Basic Usage**:
```bash
python3 scripts/tools/batch_generate_captions_yokai.py \
    /path/to/clusters \
    --cluster-analysis analysis.json
```

**Model Selection**:

**Best Quality (recommended)**:
```bash
python3 scripts/tools/batch_generate_captions_yokai.py \
    /path/to/clusters \
    --cluster-analysis analysis.json \
    --model "Salesforce/blip2-opt-6.7b" \
    --device cuda \
    --batch-size 8
```

**Fast (lower quality)**:
```bash
python3 scripts/tools/batch_generate_captions_yokai.py \
    /path/to/clusters \
    --model "Salesforce/blip2-opt-2.7b" \
    --batch-size 16
```

**CPU Mode (slow but no GPU required)**:
```bash
python3 scripts/tools/batch_generate_captions_yokai.py \
    /path/to/clusters \
    --model "Salesforce/blip2-opt-2.7b" \
    --device cpu \
    --batch-size 2
```

**Specific Clusters Only**:
```bash
python3 scripts/tools/batch_generate_captions_yokai.py \
    /path/to/clusters \
    --selected-clusters cluster_000 cluster_001 cluster_005
```

**No Trigger Words**:
```bash
python3 scripts/tools/batch_generate_captions_yokai.py \
    /path/to/clusters \
    --no-trigger-words
```

**Parameters**:
- `--model`: BLIP2 variant (2.7b, 6.7b, flan-t5-xl)
- `--device`: `cuda` or `cpu`
- `--batch-size`: Images per batch (default: 8)
- `--cluster-analysis`: Optional cluster analysis JSON
- `--selected-clusters`: Process specific clusters only
- `--no-trigger-words`: Disable automatic trigger word generation

**Features**:
- Type-specific prompts (human vs yokai)
- Automatic trigger words (char000, char001, etc.)
- Resume capability (skips existing captions)
- Batch processing

**Duration**: ~1-2 minutes per 100 images (GPU)

---

## 🗂️ Training Data Preparation

### 4. prepare_yokai_lora_training.py

**Purpose**: Organize data into kohya_ss training format

**Basic Usage**:
```bash
python3 scripts/tools/prepare_yokai_lora_training.py \
    /path/to/clusters \
    --output-dir /path/to/training_data
```

**With Analysis (recommended)**:
```bash
python3 scripts/tools/prepare_yokai_lora_training.py \
    /path/to/clusters \
    --output-dir /path/to/training_data \
    --cluster-analysis analysis.json \
    --validation-split 0.1
```

**Specific Clusters**:
```bash
python3 scripts/tools/prepare_yokai_lora_training.py \
    /path/to/clusters \
    --output-dir /path/to/training_data \
    --selected-clusters cluster_000 cluster_001 cluster_005
```

**Manual Parameters (disable auto-optimization)**:
```bash
python3 scripts/tools/prepare_yokai_lora_training.py \
    /path/to/clusters \
    --output-dir /path/to/training_data \
    --no-auto-params
```

**Parameters**:
- `--output-dir`: Output directory for training data
- `--cluster-analysis`: Cluster analysis JSON (for character types)
- `--selected-clusters`: Specific clusters to prepare
- `--validation-split`: Validation ratio (default: 0.1 = 10%)
- `--no-auto-params`: Disable automatic parameter optimization

**Output Structure**:
```
training_data/
├── 30_char_000/              # {repeat}_{name}
│   ├── image001.png
│   ├── image001.txt
│   └── ...
├── 20_char_001/
├── validation/
│   ├── char_000/
│   └── ...
├── configs/
│   ├── char_000_config.toml
│   └── ...
└── preparation_metadata.json
```

**Auto Parameter Optimization**:
| Images | Repeat | Epochs | UNet LR | Note |
|--------|--------|--------|---------|------|
| 5-10   | 30     | 30     | 5e-5    | Heavy training, low LR |
| 11-20  | 20     | 25     | 8e-5    | Moderate |
| 21-50  | 15     | 20     | 1e-4    | Standard |
| 51-100 | 10     | 15     | 1e-4    | Rich dataset |
| 100+   | 5      | 12     | 1e-4    | Minimal repetition |

---

## ✅ Validation

### 5. validate_yokai_training_data.py

**Purpose**: Validate training data before starting training

**Basic Usage**:
```bash
python3 scripts/tools/validate_yokai_training_data.py \
    /path/to/training_data
```

**With Report**:
```bash
python3 scripts/tools/validate_yokai_training_data.py \
    /path/to/training_data \
    --output-report validation_report.json
```

**Custom Thresholds**:
```bash
python3 scripts/tools/validate_yokai_training_data.py \
    /path/to/training_data \
    --output-report report.json \
    --min-resolution 512 \
    --max-aspect-ratio 3.0
```

**Parameters**:
- `--output-report`: Save validation report to JSON
- `--min-resolution`: Minimum image dimension (default: 512)
- `--max-aspect-ratio`: Maximum aspect ratio (default: 3.0)

**Checks**:
- ✅ Image integrity (readable, correct format)
- ✅ Image dimensions (≥ min resolution)
- ✅ Aspect ratio (not too extreme)
- ✅ Caption files exist and paired
- ✅ Caption quality (length, content)
- ✅ Directory structure (kohya_ss format)

**Output**:
```
Validation Summary
==================
Training Set:
  Total images: 2,156
  Valid pairs: 2,148 (99.6%)
  Invalid pairs: 8

Common Issues:
  - Caption file missing: 5
  - Resolution too low: 3

✅ VALIDATION PASSED
```

---

## 🎭 Character Selection

### 6. interactive_character_selector.py

**Purpose**: Interactively select characters for training

**Interactive Mode**:
```bash
python3 scripts/tools/interactive_character_selector.py \
    /path/to/clusters \
    --analysis analysis.json
```

**Batch Mode (non-interactive)**:

**Select S+A tier only**:
```bash
python3 scripts/tools/interactive_character_selector.py \
    /path/to/clusters \
    --analysis analysis.json \
    --batch \
    --output-dir /path/to/selected \
    --tiers S A
```

**Select by image count**:
```bash
python3 scripts/tools/interactive_character_selector.py \
    /path/to/clusters \
    --batch \
    --output-dir /path/to/selected \
    --min-images 20 \
    --max-images 100
```

**Select recommended only**:
```bash
python3 scripts/tools/interactive_character_selector.py \
    /path/to/clusters \
    --analysis analysis.json \
    --batch \
    --output-dir /path/to/selected \
    --recommended-only
```

**Select by character type**:
```bash
python3 scripts/tools/interactive_character_selector.py \
    /path/to/clusters \
    --analysis analysis.json \
    --batch \
    --output-dir /path/to/selected \
    --types yokai
```

**Interactive Commands**:
- `s` - Select/deselect current cluster
- `n` - Next cluster
- `p` - Previous cluster
- `j` - Jump to cluster number
- `v` - View sample images
- `f` - Apply filters
- `r` - Reset filters
- `a` - Auto-select all recommended
- `c` - Clear all selections
- `x` - Export selected clusters
- `q` - Quit

**Load Previous Selection**:
```bash
python3 scripts/tools/interactive_character_selector.py \
    /path/to/clusters \
    --load-selection previous_selection.json
```

---

## 🖼️ Background LoRA

### 7. prepare_background_lora.py

**Purpose**: Prepare background layers for style LoRA training

**Basic Usage**:
```bash
python3 scripts/tools/prepare_background_lora.py \
    /path/to/layered_frames/background \
    --output-dir /path/to/background_training_data
```

**Recommended Configuration**:
```bash
python3 scripts/tools/prepare_background_lora.py \
    /path/to/layered_frames/background \
    --output-dir /path/to/background_training \
    --repeat-count 10 \
    --max-backgrounds 500 \
    --similarity-threshold 0.95 \
    --device cuda
```

**High Diversity (keep more backgrounds)**:
```bash
python3 scripts/tools/prepare_background_lora.py \
    /path/to/backgrounds \
    --output-dir /path/to/output \
    --max-backgrounds 1000 \
    --similarity-threshold 0.90
```

**CPU Mode**:
```bash
python3 scripts/tools/prepare_background_lora.py \
    /path/to/backgrounds \
    --output-dir /path/to/output \
    --device cpu \
    --max-backgrounds 300
```

**Parameters**:
- `--repeat-count`: Repeat count (default: 10)
- `--validation-split`: Validation ratio (default: 0.1)
- `--max-backgrounds`: Max backgrounds to include (default: all unique)
- `--scene-type`: Scene type for captions (default: "anime")
- `--similarity-threshold`: CLIP similarity threshold (default: 0.95)
- `--device`: `cuda` or `cpu`

**Features**:
- CLIP-based duplicate filtering
- Automatic scene captioning
- Optimized training parameters for backgrounds
- TOML config generation

**Use Case**: Maintain consistent Yokai Watch art style in generated images

---

## 🔄 Common Workflows

### Workflow 1: Quick Start (Main Characters Only)

```bash
# 1. Complete pipeline with filtering
./scripts/batch/yokai_lora_complete_pipeline.sh

# 2. Select only S+A tier characters
python3 scripts/tools/interactive_character_selector.py \
    output/character_clusters \
    --analysis output/cluster_analysis.json \
    --batch \
    --output-dir output/selected_characters \
    --tiers S A

# 3. Prepare training data
python3 scripts/tools/prepare_yokai_lora_training.py \
    output/selected_characters \
    --output-dir training_data/main_characters

# 4. Validate
python3 scripts/tools/validate_yokai_training_data.py \
    training_data/main_characters

# 5. Start training
cd kohya_ss
for config in ../training_data/main_characters/configs/*.toml; do
    accelerate launch train_network.py --config_file "$config"
done
```

### Workflow 2: Small Clusters Only (Rare Characters)

```bash
# 1. Filter small clusters
python3 scripts/tools/interactive_character_selector.py \
    character_clusters \
    --batch \
    --output-dir small_clusters \
    --min-images 5 \
    --max-images 20

# 2. Heavy augmentation
python3 scripts/tools/augment_small_clusters.py \
    small_clusters \
    --output-dir augmented_small \
    --max-original 20 \
    --aug-intensity heavy

# 3. Generate captions
python3 scripts/tools/batch_generate_captions_yokai.py \
    augmented_small

# 4. Prepare training data
python3 scripts/tools/prepare_yokai_lora_training.py \
    augmented_small \
    --output-dir training_data/rare_characters
```

### Workflow 3: Manual Selection + Custom Training

```bash
# 1. Analyze clusters
python3 scripts/tools/analyze_yokai_clusters.py \
    character_clusters \
    --output-json analysis.json \
    --output-html analysis.html

# 2. Review analysis.html in browser
# 3. Select specific clusters interactively
python3 scripts/tools/interactive_character_selector.py \
    character_clusters \
    --analysis analysis.json

# 4. After selection, augment if needed
python3 scripts/tools/augment_small_clusters.py \
    exported_selection \
    --output-dir augmented_selection

# 5. Generate captions
python3 scripts/tools/batch_generate_captions_yokai.py \
    augmented_selection \
    --cluster-analysis analysis.json

# 6. Prepare with manual review
python3 scripts/tools/prepare_yokai_lora_training.py \
    augmented_selection \
    --output-dir training_data/custom_selection \
    --validation-split 0.15

# 7. Edit TOML configs manually if needed
# 8. Validate
python3 scripts/tools/validate_yokai_training_data.py \
    training_data/custom_selection

# 9. Train
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 4

# Or use CPU
--device cpu
```

### Slow Processing

```bash
# Use smaller BLIP2 model
--model "Salesforce/blip2-opt-2.7b"

# Reduce max backgrounds
--max-backgrounds 300

# Skip heavy analysis
--no-auto-params
```

### Poor Caption Quality

```bash
# Use best BLIP2 model
--model "Salesforce/blip2-opt-6.7b"

# Provide cluster analysis for better prompts
--cluster-analysis analysis.json
```

### Too Many/Few Augmentations

```bash
# Manual control
--target-multiplier 4.0  # Force 4x augmentation

# Adjust intensity
--aug-intensity light  # or heavy
```

---

## 📚 See Also

- `YOKAI_LORA_TRAINING_GUIDE.md` - Complete training guide
- `YOKAI_FULL_PROCESSING_GUIDE.md` - Segmentation & clustering guide
- `U2NET_OPTIMIZATION_REPORT.md` - Technical details

---

**Last Updated**: 2025-10-30
