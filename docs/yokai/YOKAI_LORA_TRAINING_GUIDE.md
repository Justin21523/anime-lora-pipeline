# Yokai Watch LoRA Training Complete Guide

Complete guide for training character and background LoRAs from Yokai Watch anime data.

---

## 📋 Overview

This guide covers the complete workflow from raw episode videos to trained LoRA models:

1. **Data Processing** - Segmentation & clustering (covered in `YOKAI_FULL_PROCESSING_GUIDE.md`)
2. **Cluster Analysis** - Quality assessment & character identification
3. **Data Augmentation** - Expanding small sample clusters
4. **Caption Generation** - BLIP2-based descriptions
5. **Training Data Preparation** - kohya_ss format organization
6. **LoRA Training** - Actual model training
7. **Validation & Testing** - Quality checks

---

## 🚀 Quick Start (Complete Pipeline)

### Use the Automated Pipeline

The easiest way is to run the complete pipeline script:

```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora

# Run complete pipeline (all stages)
./scripts/batch/yokai_lora_complete_pipeline.sh
```

This will:
- ✅ Process all episodes (segmentation + clustering)
- ✅ Analyze cluster quality
- ✅ Augment small clusters (5-30 images)
- ✅ Generate captions with BLIP2
- ✅ Prepare training data
- ✅ Validate data quality
- ✅ (Optional) Prepare background LoRA data

**Estimated Time**: 24-30 hours for full Yokai Watch dataset (214 episodes)

---

## 📚 Step-by-Step Guide

If you prefer manual control or need to customize specific stages:

### Prerequisites

1. **Completed Segmentation & Clustering**
   - Follow `YOKAI_FULL_PROCESSING_GUIDE.md` first
   - You should have:
     - Character layers: `layered_frames/character/`
     - Background layers: `layered_frames/background/`
     - Character clusters: `character_clusters/cluster_*/`

2. **Required Environment**
   ```bash
   conda activate blip2-env
   ```

3. **Expected Directory Structure**
   ```
   outputs/yokai_full_processing/[TIMESTAMP]/
   ├── layered_frames/
   │   ├── character/          # Character layers (RGBA)
   │   └── background/         # Background layers (RGB)
   └── character_clusters/
       ├── cluster_000/
       ├── cluster_001/
       └── ...
   ```

---

### Step 1: Cluster Analysis

Analyze cluster quality and get recommendations:

```bash
python3 scripts/tools/analyze_yokai_clusters.py \
    /path/to/character_clusters \
    --output-json cluster_analysis.json \
    --output-html cluster_analysis.html
```

**Output:**
- `cluster_analysis.json` - Machine-readable results
- `cluster_analysis.html` - Visual report (open in browser)

**What to Look For:**
- **S Tier** (100+ images, high quality): Main characters - excellent for training
- **A Tier** (50-100 images): Major characters - very good
- **B Tier** (20-50 images): Minor characters - good with augmentation
- **C Tier** (15-20 images): Rare characters - needs heavy augmentation
- **D Tier** (< 15 images): Skip or combine with similar characters

**Example Output:**
```
Cluster Analysis Summary
========================
Total clusters: 128

By Tier:
  S Tier: 5 clusters (recommended)
  A Tier: 12 clusters (recommended)
  B Tier: 25 clusters (recommended)
  C Tier: 31 clusters (requires augmentation)
  D Tier: 55 clusters (not recommended)

By Character Type:
  Human: 68 clusters
  Yokai: 52 clusters
  Unknown: 8 clusters
```

---

### Step 2: Character Selection (Optional)

Interactively select which characters to train:

```bash
python3 scripts/tools/interactive_character_selector.py \
    /path/to/character_clusters \
    --analysis cluster_analysis.json
```

**Interactive Options:**
- Browse clusters one by one
- View sample images
- Filter by quality, tier, type, size
- Auto-select all recommended
- Export selected to new directory

**Batch Selection (Non-Interactive):**
```bash
# Select only S+A tier clusters
python3 scripts/tools/interactive_character_selector.py \
    /path/to/character_clusters \
    --analysis cluster_analysis.json \
    --batch \
    --output-dir /path/to/selected_clusters \
    --tiers S A \
    --min-images 20
```

---

### Step 3: Data Augmentation

Augment small clusters to have enough training samples:

```bash
python3 scripts/tools/augment_small_clusters.py \
    /path/to/character_clusters \
    --output-dir /path/to/augmented_clusters \
    --max-original 30 \
    --min-original 5 \
    --aug-intensity medium
```

**Parameters:**
- `--max-original 30`: Only augment clusters with ≤30 images
- `--min-original 5`: Minimum images required (skip smaller clusters)
- `--aug-intensity`: `light`, `medium`, or `heavy`

**Augmentation Multipliers (Auto):**
- 5-10 images → 6-8x augmentation (final: 40-60 images)
- 11-15 images → 4-5x augmentation (final: 50-70 images)
- 16-20 images → 3-4x augmentation (final: 60-80 images)
- 21-30 images → 1.5-2x augmentation

**Augmentation Techniques:**
- Horizontal flip (for symmetric characters)
- Slight rotation (-5° to +5°)
- Brightness adjustment (0.8-1.2x)
- Contrast adjustment (0.9-1.1x)
- Hue shift (±10°, preserving character colors)
- Crop variation (95-100%)
- Optional slight blur

**Important:** Augmentation preserves character identity while adding variety.

---

### Step 4: Caption Generation

Generate training captions using BLIP2:

```bash
python3 scripts/tools/batch_generate_captions_yokai.py \
    /path/to/augmented_clusters \
    --cluster-analysis cluster_analysis.json \
    --model "Salesforce/blip2-opt-6.7b" \
    --device cuda \
    --batch-size 8
```

**Parameters:**
- `--model`: BLIP2 model variant
  - `blip2-opt-2.7b` - Faster, less detailed
  - `blip2-opt-6.7b` - **Recommended** - Best quality
  - `blip2-flan-t5-xl` - Alternative, good quality
- `--device`: `cuda` or `cpu`
- `--batch-size`: Higher = faster but more VRAM (default: 8)

**Features:**
- **Type-specific prompts**: Different prompts for humans vs yokai
- **Automatic trigger words**: Prepends cluster-based trigger (e.g., "char000")
- **Resume capability**: Skips images that already have captions

**Expected Time:**
- ~1-2 minutes per 100 images (GPU)
- For 100 clusters with 50 images each: ~30-60 minutes

**Example Captions:**
```
Human character:
"char042, a young anime boy with brown hair, wearing a blue soccer uniform,
smiling expression, standing pose, detailed clothing"

Yokai character:
"char017, a red cat-like yokai with two tails, distinctive yellow eyes,
cheerful expression, unique fire pattern on forehead"
```

---

### Step 5: Training Data Preparation

Organize data into kohya_ss training format:

```bash
python3 scripts/tools/prepare_yokai_lora_training.py \
    /path/to/augmented_clusters \
    --output-dir /path/to/training_data \
    --cluster-analysis cluster_analysis.json \
    --validation-split 0.1
```

**Parameters:**
- `--validation-split`: Validation set ratio (default: 0.1 = 10%)
- `--selected-clusters`: Specific clusters to prepare (optional)
- `--no-auto-params`: Disable automatic parameter optimization

**What It Does:**
1. Creates kohya_ss directory structure:
   ```
   training_data/
   ├── 30_char_000/              # {repeat_count}_{character_name}
   │   ├── image001.png
   │   ├── image001.txt
   │   └── ...
   ├── 20_char_001/
   │   └── ...
   ├── validation/
   │   ├── char_000/
   │   └── ...
   └── configs/
       ├── char_000_config.toml
       └── ...
   ```

2. **Automatic parameter optimization** based on sample count:

   | Images | Repeat | Epochs | UNet LR | Text LR | Strategy |
   |--------|--------|--------|---------|---------|----------|
   | 5-10   | 30     | 30     | 5e-5    | 2e-5    | Heavy repetition, low LR |
   | 11-20  | 20     | 25     | 8e-5    | 3e-5    | Moderate repetition |
   | 21-50  | 15     | 20     | 1e-4    | 5e-5    | Standard training |
   | 51-100 | 10     | 15     | 1e-4    | 5e-5    | Rich dataset |
   | 100+   | 5      | 12     | 1e-4    | 5e-5    | Minimal repetition |

3. **Character type adjustments**:
   - Yokai: Higher network_dim (48 vs 32) for complex features
   - Human: Standard parameters

4. Generates TOML config files for each character

**Output:**
```
Preparation Complete
====================
  Characters prepared: 42
  Training images: 2,156
  Validation images: 239
  Output: /path/to/training_data

Ready to train! Example commands:
1. char_000:
   accelerate launch train_network.py --config_file /path/to/configs/char_000_config.toml
```

---

### Step 6: Validation

Validate training data before starting training:

```bash
python3 scripts/tools/validate_yokai_training_data.py \
    /path/to/training_data \
    --output-report validation_report.json \
    --min-resolution 512
```

**Checks:**
- ✅ Image integrity (readable, correct format)
- ✅ Image quality (resolution, aspect ratio)
- ✅ Caption files exist and paired correctly
- ✅ Directory structure (kohya_ss format)
- ✅ Filename conventions

**Example Output:**
```
Validation Summary
==================
Training Set:
  Total images: 2,156
  Valid pairs: 2,148 (99.6%)
  Invalid pairs: 8
  Pairs with warnings: 23

Common Issues:
  - Caption file missing: 5 occurrences
  - Resolution too low: 3 occurrences

Common Warnings:
  - Very short caption: 15 occurrences
  - Extreme aspect ratio: 8 occurrences

✅ VALIDATION PASSED - Training data is ready
```

**Fix Issues:**
- Delete invalid images or regenerate captions
- Re-run validation until 100% valid

---

### Step 7: LoRA Training

#### Setup kohya_ss (if not already installed)

```bash
# Clone kohya_ss
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

# Install dependencies
pip install -r requirements.txt
pip install -U -I --no-deps xformers
```

#### Start Training

```bash
cd /path/to/kohya_ss

# Train single character
accelerate launch train_network.py \
    --config_file /path/to/training_data/configs/char_000_config.toml

# Monitor with tensorboard
tensorboard --logdir ./logs
```

#### Training Multiple Characters in Parallel

```bash
# Use GNU parallel or tmux
for config in /path/to/training_data/configs/*.toml; do
    tmux new-session -d -s "train_$(basename $config .toml)" \
        "accelerate launch train_network.py --config_file $config"
done

# Monitor all sessions
tmux ls
```

#### Expected Training Times

| Dataset Size | Epochs | Repeat | GPU (RTX 3090) | GPU (RTX 4090) |
|--------------|--------|--------|----------------|----------------|
| 10 images    | 30     | 30     | ~30 min        | ~20 min        |
| 25 images    | 25     | 20     | ~1 hour        | ~40 min        |
| 50 images    | 20     | 15     | ~1.5 hours     | ~1 hour        |
| 100 images   | 15     | 10     | ~2 hours       | ~1.5 hours     |

**Tips:**
- Monitor loss curves in tensorboard
- Save every 2-3 epochs to compare checkpoints
- Lower learning rate if loss is unstable
- Increase learning rate if loss plateaus early

---

### Step 8: Background LoRA (Optional)

Train a background LoRA for consistent art style:

```bash
python3 scripts/tools/prepare_background_lora.py \
    /path/to/layered_frames/background \
    --output-dir /path/to/background_training_data \
    --repeat-count 10 \
    --max-backgrounds 500 \
    --similarity-threshold 0.95
```

**Parameters:**
- `--max-backgrounds`: Limit to N most diverse backgrounds
- `--similarity-threshold`: CLIP similarity for duplicate filtering (0.95 = keep 5% different)

**Features:**
- **CLIP-based deduplication**: Removes near-identical backgrounds
- **Automatic captioning**: Simple scene-based descriptions
- **Lower learning rates**: Backgrounds need gentler training

**Use Case:**
- Generate new character poses in authentic Yokai Watch environments
- Maintain consistent art style and color palette
- Combine with character LoRAs for complete scene generation

---

## 🎯 Training Tips & Best Practices

### For Small Datasets (5-20 images)

✅ **Do:**
- Use data augmentation (6-8x multiplier)
- Higher repeat counts (20-30)
- Lower learning rates (5e-5 to 8e-5)
- More epochs (25-30)
- Monitor for overfitting

❌ **Don't:**
- Skip augmentation
- Use high learning rates
- Train too few epochs
- Ignore validation loss

### For Large Datasets (50+ images)

✅ **Do:**
- Minimal or no augmentation
- Standard learning rates (1e-4)
- Moderate epochs (12-20)
- Focus on diversity in training data

❌ **Don't:**
- Over-augment (causes style degradation)
- Use excessive repeat counts
- Train too many epochs

### For Yokai Characters

✅ **Do:**
- Use higher network_dim (48)
- Focus caption on distinctive features (colors, shapes, patterns)
- Include body type and unique characteristics
- Slightly higher learning rate (1.1x)

### For Human Characters

✅ **Do:**
- Standard network_dim (32)
- Focus caption on clothing, hair, facial features
- Include emotional expressions
- Standard learning rates

### General Best Practices

1. **Always validate data** before training
2. **Start with recommended characters** (S/A tier)
3. **Train small sample first** (1-2 characters) to verify pipeline
4. **Monitor training logs** for loss curves
5. **Test multiple checkpoints** to find best epoch
6. **Use validation set** to detect overfitting early
7. **Save checkpoint every 2 epochs** for comparison

---

## 📊 Quality Evaluation

### After Training

1. **Generate test images**:
   ```bash
   # Use your trained LoRA with trigger word
   prompt: "char000, standing pose, neutral background"
   ```

2. **Evaluate quality**:
   - ✅ Character features accurate
   - ✅ Consistent appearance across poses
   - ✅ No artifacts or distortions
   - ✅ Responds well to different prompts

3. **Common Issues & Fixes**:

   | Issue | Cause | Fix |
   |-------|-------|-----|
   | Blurry output | Overtraining | Use earlier checkpoint |
   | Inconsistent features | Undertrained | Train more epochs |
   | Wrong colors | Caption mismatch | Regenerate captions |
   | Artifacts | Too high LR | Reduce learning rate |
   | Style bleed | Mixed data | Filter clusters better |

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solutions:**
```bash
# Reduce batch size
--batch-size 4

# Use gradient checkpointing (already enabled in configs)

# Use CPU for CLIP/BLIP2 (slower)
--device cpu
```

### Issue: Captions Too Generic

**Solution:**
```bash
# Use larger BLIP2 model
--model "Salesforce/blip2-opt-6.7b"  # Best quality

# Or manually edit important character captions
```

### Issue: Training Loss Not Decreasing

**Causes:**
- Learning rate too high → Reduce by 0.5x
- Learning rate too low → Increase by 2x
- Bad data quality → Re-filter clusters
- Wrong trigger word format → Check captions

### Issue: Validation Loss Increasing (Overfitting)

**Solutions:**
- Stop training earlier (use earlier checkpoint)
- Reduce repeat count
- Add more diverse training data
- Enable dropout (advanced)

---

## 📁 File Organization

Recommended project structure:

```
yokai_lora_project/
├── raw_data/
│   └── episodes/              # Original video files
├── processed/
│   ├── layered_frames/        # Segmentation output
│   └── character_clusters/    # Clustering output
├── analysis/
│   ├── cluster_analysis.json
│   └── cluster_analysis.html
├── training_data/
│   ├── characters/            # Prepared character training data
│   │   ├── 30_char_000/
│   │   ├── validation/
│   │   └── configs/
│   └── backgrounds/           # Prepared background training data
├── trained_models/
│   ├── char_000/
│   │   ├── char_000_lora-000010.safetensors
│   │   └── logs/
│   └── backgrounds/
└── logs/
    ├── pipeline.log
    └── validation_report.json
```

---

## 🚀 Advanced Usage

### Batch Training All Characters

```bash
# Create batch training script
cat > train_all.sh <<'EOF'
#!/bin/bash
for config in training_data/configs/*.toml; do
    char_name=$(basename $config .toml)
    echo "Training $char_name..."

    tmux new-session -d -s "train_$char_name" \
        "accelerate launch train_network.py --config_file $config 2>&1 | tee logs/train_${char_name}.log"
done
EOF

chmod +x train_all.sh
./train_all.sh
```

### Custom Training Parameters

Edit generated TOML configs:

```toml
# Increase network capacity for complex characters
network_dim = 64
network_alpha = 32

# Adjust learning rates
unet_lr = 1.2e-4
text_encoder_lr = 6e-5

# Change batch size
train_batch_size = 2  # Lower if OOM

# Add noise offset (improves dark/light scene handling)
noise_offset = 0.1
```

### Multi-Concept Training

Train multiple characters in one LoRA:

```bash
# Prepare combined training data
mkdir -p multi_concept_training
cp -r training_data/30_char_000 multi_concept_training/
cp -r training_data/25_char_001 multi_concept_training/
cp -r training_data/20_char_002 multi_concept_training/

# Create combined config
# Edit config to point to multi_concept_training directory
```

---

## 📚 Additional Resources

- **kohya_ss Documentation**: https://github.com/kohya-ss/sd-scripts
- **LoRA Training Guide**: https://rentry.org/lora_train
- **BLIP2 Paper**: https://arxiv.org/abs/2301.12597

---

## ✅ Checklist

### Before Training

- [ ] Completed segmentation & clustering
- [ ] Analyzed cluster quality (HTML report)
- [ ] Selected characters to train (S/A/B tier recommended)
- [ ] Augmented small clusters (< 30 images)
- [ ] Generated captions for all images
- [ ] Prepared training data (kohya_ss format)
- [ ] Validated training data (100% valid)
- [ ] Reviewed training configs
- [ ] Set up kohya_ss environment

### During Training

- [ ] Monitor tensorboard logs
- [ ] Check loss curves every few epochs
- [ ] Verify checkpoint quality
- [ ] Watch for overfitting (validation loss)
- [ ] Note best performing epoch

### After Training

- [ ] Test trained LoRA with various prompts
- [ ] Compare multiple checkpoint epochs
- [ ] Verify character consistency
- [ ] Check for artifacts or issues
- [ ] Document trigger words and usage
- [ ] Archive best checkpoints

---

**Last Updated**: 2025-10-30

**Related Documentation**:
- `YOKAI_FULL_PROCESSING_GUIDE.md` - Complete dataset processing
- `U2NET_OPTIMIZATION_REPORT.md` - Segmentation optimization details
- `PROJECT_STATUS.md` - Overall project status
