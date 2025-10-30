# Yokai Watch Full Dataset Processing Guide

Complete guide for processing the entire Yokai Watch dataset with optimized parameters.

---

## 📋 Overview

The `process_yokai_full_optimized.sh` script automates the complete processing pipeline:

1. **Stage 1: Layered Segmentation** - U2Net with "Aggressive" refinement
2. **Stage 2: Character Clustering** - Optimized HDBSCAN (detects 100+ characters)

### What You Get

After processing:
- ✅ Character layers extracted from all episodes
- ✅ Background layers (for potential background LoRA training)
- ✅ Automatic character clustering (humans + yokai)
- ✅ Complete logs and statistics
- ✅ Ready-to-use training data

---

## 🚀 Quick Start

### Basic Usage (Default Settings)

```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora
./scripts/batch/process_yokai_full_optimized.sh
```

This will:
- Read episodes from: `/home/b0979/yokai_input_fast`
- Output to: `/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai_full_processing/[TIMESTAMP]`
- Use 16 parallel workers for segmentation
- Use optimized clustering parameters

### Custom Configuration

Override any parameter using environment variables:

```bash
# Custom input/output directories
INPUT_DIR=/path/to/episodes \
OUTPUT_DIR=/path/to/output \
./scripts/batch/process_yokai_full_optimized.sh

# Adjust worker counts for your system
SEGMENTATION_WORKERS=8 \
CLUSTERING_WORKERS=4 \
./scripts/batch/process_yokai_full_optimized.sh

# Use CPU instead of GPU for clustering
CLUSTERING_DEVICE=cpu \
./scripts/batch/process_yokai_full_optimized.sh

# Change minimum cluster size
CLUSTERING_MIN_SIZE=15 \
./scripts/batch/process_yokai_full_optimized.sh
```

---

## ⚙️ Configuration Parameters

### Directory Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/home/b0979/yokai_input_fast` | Directory containing episode folders (S1.01, S1.02, etc.) |
| `OUTPUT_BASE` | `/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai_full_processing` | Base output directory |
| `OUTPUT_DIR` | `${OUTPUT_BASE}/[TIMESTAMP]` | Final output directory (auto-timestamped) |

### Stage 1: Segmentation

| Variable | Default | Description |
|----------|---------|-------------|
| `SEGMENTATION_WORKERS` | `16` | Parallel workers (CPU cores) |
| `SEGMENTATION_MODEL` | `u2net` | Model type (u2net, u2netp, isnet-anime) |
| `INPAINT_METHOD` | `telea` | Background inpainting (telea, lama) |

**Recommended**: Keep defaults. U2Net with 16 workers provides best balance.

### Stage 2: Clustering

| Variable | Default | Description |
|----------|---------|-------------|
| `CLUSTERING_MIN_SIZE` | `25` | Minimum images per cluster (lower = more clusters) |
| `CLUSTERING_BATCH_SIZE` | `64` | CLIP batch size (higher = faster but more VRAM) |
| `CLUSTERING_WORKERS` | `8` | DataLoader workers |
| `CLUSTERING_DEVICE` | `cuda` | Processing device (cuda/cpu) |

**Notes**:
- `CLUSTERING_MIN_SIZE=25` with optimized HDBSCAN detects ~100-150 clusters
- Lower to `15-20` to detect more minor characters
- `CLUSTERING_DEVICE=cpu` if you encounter VRAM issues

### Other Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RESUME_FROM` | (empty) | Resume from specific episode (e.g., `S1.05`) |
| `CONDA_ENV` | `blip2-env` | Conda environment name |

---

## 🔄 Resume Capability

If processing is interrupted, you can resume from a specific episode:

```bash
RESUME_FROM=S1.15 ./scripts/batch/process_yokai_full_optimized.sh
```

This will skip already-processed episodes and continue from S1.15 onwards.

**Note**: The script automatically marks completed episodes with `.done` files, so it won't re-process them.

---

## 📊 Expected Output

### Directory Structure

```
outputs/yokai_full_processing/[TIMESTAMP]/
├── config.txt                           # Configuration used
├── processing_report.txt                # Final statistics
├── layered_frames/
│   ├── character/                       # Character layers (RGBA)
│   │   ├── scene0001_character.png
│   │   ├── scene0002_character.png
│   │   └── ...
│   ├── background/                      # Background layers (RGB)
│   │   ├── scene0001_background.jpg
│   │   └── ...
│   └── masks/                           # Segmentation masks
│       ├── scene0001_mask.png
│       └── ...
├── character_clusters/
│   ├── cluster_000/                     # 11 images
│   ├── cluster_001/                     # 17 images
│   ├── ...
│   ├── cluster_123/                     # 144 images (likely main character)
│   └── clustering_metadata.json
└── logs/
    ├── segmentation_S1.01.log
    ├── segmentation_S1.02.log
    ├── ...
    └── clustering_full.log
```

### Performance Estimates

Based on test results (3 episodes, 3,752 frames):

| Metric | 3 Episodes | 50 Episodes (estimated) | 100 Episodes (estimated) |
|--------|-----------|------------------------|--------------------------|
| Character Layers | 3,752 | ~62,500 | ~125,000 |
| Segmentation Time | ~20 min | ~5.5 hours | ~11 hours |
| Clustering Time | ~2 min | ~15 min | ~30 min |
| **Total Time** | **~22 min** | **~6 hours** | **~12 hours** |
| Character Clusters | 128 | ~200-250 | ~250-300 |

**Note**: Actual numbers depend on episode lengths and content variety.

---

## 📈 Monitoring Progress

### During Processing

The script provides real-time progress:

```
========================================
Episode 15/50: S1.15
========================================
Processing: 45% |████████      | 2250/5000 [02:15<02:45, 16.7img/s]
```

### Check Logs

```bash
# View segmentation log for specific episode
tail -f outputs/yokai_full_processing/[TIMESTAMP]/logs/segmentation_S1.15.log

# View clustering log
tail -f outputs/yokai_full_processing/[TIMESTAMP]/logs/clustering_full.log
```

### Quick Statistics

```bash
# Count processed character layers
find outputs/yokai_full_processing/[TIMESTAMP]/layered_frames/character/ -name "*.png" | wc -l

# Count detected clusters
ls -d outputs/yokai_full_processing/[TIMESTAMP]/character_clusters/cluster_* | wc -l

# View largest clusters
for d in outputs/yokai_full_processing/[TIMESTAMP]/character_clusters/cluster_*; do
    echo "$(ls -1 $d/*.png | wc -l) $(basename $d)"
done | sort -rn | head -20
```

---

## 🎯 After Processing Complete

### 1. Review Character Clusters

Open the output directory in Windows Explorer:

```bash
explorer.exe $(wslpath -w "outputs/yokai_full_processing/[TIMESTAMP]/character_clusters")
```

Or use a file browser to visually inspect:
- Which clusters are main characters
- Which are yokai
- Quality of clustering

### 2. Select Characters for Training

Recommended selection criteria:
- **Main characters**: Clusters with 50+ images
- **Secondary characters**: Clusters with 20-50 images
- **Yokai**: Clusters with 15+ images

Skip clusters with:
- < 15 images (too few for training)
- Mixed characters (check if clustering needs adjustment)
- Low quality or unclear images

### 3. Next Steps

See `LORA_TRAINING_GUIDE.md` (to be created) for:
- Preparing training data from clusters
- Generating captions with BLIP2
- Running LoRA training
- Evaluating results

---

## 🛠️ Troubleshooting

### Issue: "No episodes found"

**Cause**: Input directory doesn't contain episode folders named like `S1.01`, `S1.02`, etc.

**Solution**:
```bash
# Check your input directory structure
ls -la /home/b0979/yokai_input_fast/

# Should see:
# S1.01/
# S1.02/
# ...
```

### Issue: Out of memory during clustering

**Cause**: GPU VRAM exhausted with large batch size

**Solutions**:
```bash
# Reduce batch size
CLUSTERING_BATCH_SIZE=32 ./scripts/batch/process_yokai_full_optimized.sh

# Or use CPU (slower but more memory)
CLUSTERING_DEVICE=cpu ./scripts/batch/process_yokai_full_optimized.sh
```

### Issue: Segmentation is slow

**Cause**: Too few workers or CPU bottleneck

**Solutions**:
```bash
# Check CPU core count
nproc

# Adjust workers (leave 2 cores for system)
SEGMENTATION_WORKERS=$(($(nproc) - 2)) ./scripts/batch/process_yokai_full_optimized.sh
```

### Issue: Too many/too few clusters

**Adjust clustering sensitivity**:

```bash
# MORE clusters (detect minor characters)
CLUSTERING_MIN_SIZE=15 ./scripts/batch/process_yokai_full_optimized.sh

# FEWER clusters (only main characters)
CLUSTERING_MIN_SIZE=40 ./scripts/batch/process_yokai_full_optimized.sh
```

**Note**: Current optimized parameters (`min_cluster_size=25` with HDBSCAN tweaks) should detect ~100-150 clusters, which is ideal for Yokai Watch.

### Issue: Processing interrupted

**Resume from where you left off**:

```bash
# Find last processed episode
ls outputs/yokai_full_processing/[TIMESTAMP]/layered_frames/*.done | tail -1

# Resume from next episode
RESUME_FROM=S1.16 ./scripts/batch/process_yokai_full_optimized.sh
```

---

## 📝 Advanced Usage

### Running in Background (tmux)

For long-running processing:

```bash
# Start tmux session
tmux new -s yokai_processing

# Run script
./scripts/batch/process_yokai_full_optimized.sh

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t yokai_processing
```

### Custom Output Organization

```bash
# Process by season
for season in 1 2 3; do
    INPUT_DIR=/home/b0979/yokai_season${season} \
    OUTPUT_DIR=/mnt/c/outputs/yokai_season${season} \
    ./scripts/batch/process_yokai_full_optimized.sh
done
```

### Dry Run (Check Configuration)

```bash
# The script will show configuration and stop at first episode
# Just Ctrl+C after seeing the config
./scripts/batch/process_yokai_full_optimized.sh
```

---

## 📊 Performance Tuning

### Optimize for Your Hardware

**High-end System** (32+ cores, 16GB+ VRAM):
```bash
SEGMENTATION_WORKERS=24 \
CLUSTERING_BATCH_SIZE=128 \
CLUSTERING_WORKERS=12 \
./scripts/batch/process_yokai_full_optimized.sh
```

**Mid-range System** (16 cores, 8GB VRAM):
```bash
SEGMENTATION_WORKERS=12 \
CLUSTERING_BATCH_SIZE=64 \
CLUSTERING_WORKERS=6 \
./scripts/batch/process_yokai_full_optimized.sh
```

**Low-end System** (8 cores, 4GB VRAM or CPU only):
```bash
SEGMENTATION_WORKERS=6 \
CLUSTERING_BATCH_SIZE=32 \
CLUSTERING_WORKERS=4 \
CLUSTERING_DEVICE=cpu \
./scripts/batch/process_yokai_full_optimized.sh
```

---

## 🔍 Quality Validation

After processing, validate results:

### 1. Check Segmentation Quality

```bash
# Random sample of character layers
OUTPUT_DIR="outputs/yokai_full_processing/[TIMESTAMP]"
find "${OUTPUT_DIR}/layered_frames/character/" -name "*.png" | shuf | head -10
```

Open these in an image viewer - characters should be:
- ✅ Complete (no missing body parts)
- ✅ Clean edges
- ✅ Transparent background
- ❌ Not cut off or incomplete

### 2. Check Clustering Quality

```bash
# View samples from top 10 clusters
OUTPUT_DIR="outputs/yokai_full_processing/[TIMESTAMP]"
for cluster in ${OUTPUT_DIR}/character_clusters/cluster_{000..009}; do
    echo "=== $(basename $cluster) ==="
    ls "$cluster"/*.png | head -5
done
```

Each cluster should contain:
- ✅ Same character across all images
- ✅ Various poses/expressions
- ❌ No mixed characters

### 3. Statistics Validation

```bash
# Should see reasonable numbers
cat outputs/yokai_full_processing/[TIMESTAMP]/processing_report.txt
```

Expected ranges:
- **Character Layers**: 50,000 - 150,000 (depends on episode count)
- **Clusters**: 100 - 300 (Yokai Watch has many characters)
- **Clustered Rate**: 60-80% (20-40% noise is normal)

---

## 📚 Related Documentation

- `U2NET_OPTIMIZATION_REPORT.md` - Segmentation parameter optimization details
- `SEGMENTATION_MODEL_EVALUATION_REPORT.md` - Model comparison study
- `LORA_TRAINING_GUIDE.md` - Next steps after processing (to be created)

---

## ✅ Checklist

Before running full processing:

- [ ] Verify input directory contains episode folders (S1.01, S1.02, ...)
- [ ] Check available disk space (estimate: ~100GB per 50 episodes)
- [ ] Ensure conda environment `blip2-env` is set up
- [ ] Test on 1-2 episodes first (use test script)
- [ ] Decide if running in background (tmux recommended)
- [ ] Note expected duration (~6-12 hours for full dataset)

After processing:

- [ ] Review processing report
- [ ] Visually inspect 10-20 random character layers
- [ ] Check top 10 largest clusters
- [ ] Verify cluster count is reasonable (100-300)
- [ ] Select characters for LoRA training
- [ ] Proceed to caption generation

---

## 🆘 Support

If you encounter issues:

1. Check logs in `outputs/yokai_full_processing/[TIMESTAMP]/logs/`
2. Review this troubleshooting guide
3. Check GPU/memory usage: `nvidia-smi`, `htop`
4. Verify disk space: `df -h`

For optimization questions, refer to `U2NET_OPTIMIZATION_REPORT.md`.

---

**Last Updated**: 2025-10-30
**Script Version**: 1.0 (Optimized)
