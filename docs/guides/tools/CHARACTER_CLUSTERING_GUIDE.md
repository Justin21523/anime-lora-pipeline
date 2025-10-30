# Character Clustering Guide

Comprehensive guide for automatically identifying and organizing anime characters from segmented frames.

## Overview

Character clustering uses **CLIP embeddings** and **HDBSCAN clustering** to automatically group similar characters together, enabling:

- **Automatic character identification** - No manual labeling required
- **Quality filtering** - Remove blurry, small, or low-quality images
- **Organized datasets** - Each character gets their own folder
- **Training-ready structure** - Direct input for LoRA training

---

## Quick Start

### Basic Usage

```bash
# Cluster all characters from layered frames
conda run -n blip2-env python3 scripts/tools/character_clustering.py \
  /path/to/layered_frames \
  --output-dir /path/to/clustered_characters
```

This will:
1. Scan all `episode_*/character/` directories
2. Extract CLIP features from each image
3. Filter out low-quality images
4. Cluster characters automatically
5. Organize images into `character_XXX/` folders

### Output Structure

```
clustered_characters/
├── character_000/        # Main character (e.g., Endou Mamoru)
│   ├── scene0001_frame000012_t48.50s_character.png
│   ├── scene0002_frame000087_t349.00s_character.png
│   └── ... (300+ images)
├── character_001/        # Second character (e.g., Gouenji Shuuya)
│   └── ... (250+ images)
├── character_002/        # Third character
│   └── ... (180+ images)
├── noise/                # Outliers (multiple different characters)
│   └── ... (rejected/misc images)
├── cluster_visualization.png      # PCA visualization
├── clustering_summary.json        # Statistics
└── quality_report.json            # Quality filtering details
```

---

## How It Works

### 1. Feature Extraction (CLIP)

CLIP is a vision-language model that generates semantic embeddings for images.

```python
# For each character image:
Image → CLIP encoder → 512-dim feature vector
```

**Why CLIP?**
- Captures visual similarity (pose, clothing, face)
- Robust to variations (angle, lighting, scale)
- Pre-trained on diverse data

### 2. Quality Filtering

Before clustering, filter out low-quality images:

| Filter | Criteria | Purpose |
|--------|----------|---------|
| **Size** | Min 64x64 pixels | Remove tiny/cropped characters |
| **Blur** | Laplacian variance > 100 | Remove motion blur |
| **Coverage** | Alpha > 5% of pixels | Remove mostly-transparent |

**Example**: From 16,929 frames → ~15,000 high-quality images

### 3. Clustering (HDBSCAN)

**HDBSCAN** (Hierarchical Density-Based Spatial Clustering) automatically finds character groups:

- **No need to specify number of clusters**
- **Handles noise** (outliers marked as `-1`)
- **Variable cluster sizes** (main characters get more frames)

**Parameters:**
- `min_cluster_size`: Minimum images per character (default: 10)
- `min_samples`: Core density requirement (default: 5)

**Result**: Characters automatically separated by visual similarity!

### 4. Organization

Images organized into folders:

```bash
character_000/  # Largest cluster (likely protagonist)
character_001/  # Second largest
...
character_N/    # Smaller recurring characters
noise/          # Non-recurring or mixed
```

---

## Command-Line Options

### Basic Options

```bash
python3 scripts/tools/character_clustering.py INPUT_DIR \
  --output-dir OUTPUT_DIR \
  [OPTIONS]
```

### Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `ViT-B/32` | CLIP model (`ViT-B/32`, `ViT-B/16`, `ViT-L/14`) |
| `--min-cluster-size` | `10` | Minimum frames per character |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--copy` | symlinks | Copy files instead of symlinking |
| `--no-visualize` | False | Skip PCA visualization |

### Examples

**High-quality clusters (larger minimum)**:
```bash
python3 character_clustering.py /path/to/frames \
  --output-dir /path/to/output \
  --min-cluster-size 20
```

**Better embeddings (slower)**:
```bash
python3 character_clustering.py /path/to/frames \
  --output-dir /path/to/output \
  --model ViT-L/14
```

**Copy files instead of symlinks**:
```bash
python3 character_clustering.py /path/to/frames \
  --output-dir /path/to/output \
  --copy
```

---

## Performance

### Processing Time

| Dataset Size | GPU (RTX 3080) | CPU (16-core) |
|--------------|----------------|---------------|
| 1,000 images | ~2-3 min | ~10-15 min |
| 5,000 images | ~8-12 min | ~40-60 min |
| 16,929 images | ~25-35 min | ~2-3 hours |

**Bottlenecks:**
1. CLIP feature extraction (~80% of time)
2. Quality filtering (~10%)
3. HDBSCAN clustering (~10%)

### Storage

**Symlinks** (default):
- Negligible additional storage
- Fast organization
- Original files preserved

**Copy mode**:
- Duplicates all files
- ~8GB for 16,929 frames
- Useful for separate processing

---

## Understanding Results

### Cluster Visualization

`cluster_visualization.png` shows a 2D PCA projection:

```
PC1 vs PC2 scatter plot
- Each point = one character image
- Color = cluster ID
- Gray X = noise/outliers
```

**What to look for:**
- **Tight clusters**: Well-defined characters
- **Scattered clusters**: Mixed poses/angles
- **Large noise**: May need lower `min_cluster_size`

### Clustering Summary

`clustering_summary.json`:

```json
{
  "timestamp": "2025-10-27T12:00:00",
  "organization": {
    "total_images": 15234,
    "clusters": {
      "0": 487,  // Character 0: 487 images
      "1": 352,  // Character 1: 352 images
      "2": 298,  // Character 2: 298 images
      ...
    },
    "noise": 1247
  },
  "min_cluster_size": 15
}
```

**Interpreting:**
- **Large clusters** (300+ images): Main characters
- **Medium clusters** (50-200): Supporting characters
- **Small clusters** (15-50): Recurring minor characters
- **Noise**: One-off appearances, crowd scenes

### Quality Report

`quality_report.json` shows filtering results:

```json
{
  "path": "episode_001/character/frame_00123.png",
  "quality": {
    "passed": true,
    "width": 512,
    "height": 768,
    "blur_score": 234.5,  // Higher = sharper
    "alpha_coverage": 0.45,  // 45% non-transparent
    "reasons": {
      "size": true,
      "blur": true,
      "content": true
    }
  }
}
```

---

## Advanced Usage

### Batch Processing Script

Process all episodes sequentially:

```bash
#!/bin/bash
FRAMES_DIR="/path/to/layered_frames"
OUTPUT_DIR="/path/to/clustered_output"

for episode in "$FRAMES_DIR"/episode_*; do
    echo "Processing $(basename $episode)..."

    python3 scripts/tools/character_clustering.py \
        "$episode" \
        --output-dir "$OUTPUT_DIR/$(basename $episode)" \
        --min-cluster-size 15
done
```

### Multi-Episode Clustering

Cluster ALL episodes together (recommended):

```bash
# Create directory with all episodes
mkdir -p /tmp/all_episodes
for ep in /path/to/layered_frames/episode_*; do
    ln -sf "$ep" "/tmp/all_episodes/$(basename $ep)"
done

# Cluster across all episodes
python3 character_clustering.py /tmp/all_episodes \
  --output-dir /path/to/all_characters \
  --min-cluster-size 25
```

**Why cluster all together?**
- Characters appear across multiple episodes
- Better clustering with more data
- Consistent character IDs

### Fine-Tuning Parameters

**Too many clusters?** → Increase `min_cluster_size`:
```bash
--min-cluster-size 30  # Only characters with 30+ appearances
```

**Missing characters?** → Decrease `min_cluster_size`:
```bash
--min-cluster-size 5  # Include minor characters
```

**Better embeddings?** → Use larger CLIP model:
```bash
--model ViT-L/14  # More accurate, but slower
```

---

## Integration with LoRA Training

### Step 1: Cluster Characters

```bash
python3 character_clustering.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/layered_frames \
  --output-dir /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/clustered
```

### Step 2: Identify Characters

Review `cluster_visualization.png` and sample images:

```bash
# View random samples from each cluster
for cluster in clustered/character_*; do
    echo "=== $(basename $cluster) ==="
    ls "$cluster" | shuf | head -5
done
```

### Step 3: Rename Clusters

```bash
# Rename based on character identification
mv clustered/character_000 clustered/endou_mamoru
mv clustered/character_001 clustered/gouenji_shuuya
mv clustered/character_002 clustered/kidou_yuuto
```

### Step 4: Train LoRA

```bash
# Train on specific character
python3 scripts/training/train_character_lora.py \
  --character_name "endou_mamoru" \
  --input_images /path/to/clustered/endou_mamoru \
  --output_dir /path/to/lora_models/
```

---

## Troubleshooting

### Issue: Too Many Small Clusters

**Symptoms**: 50+ clusters, many with <20 images

**Solutions**:
1. Increase `min_cluster_size`:
   ```bash
   --min-cluster-size 25
   ```

2. Check quality filtering - may be too strict

### Issue: Characters Mixed in Same Cluster

**Symptoms**: Multiple different characters in one cluster

**Causes**:
- Similar clothing/uniforms
- Similar poses
- Limited training data

**Solutions**:
1. Use larger CLIP model:
   ```bash
   --model ViT-L/14
   ```

2. Manual separation post-clustering

3. Use additional metadata (episode, scene)

### Issue: Main Character Not Largest Cluster

**Symptoms**: Protagonist has fewer images than supporting cast

**Causes**:
- Quality filtering removed many frames
- Protagonist has more varied appearances

**Solutions**:
1. Check `quality_report.json` for filtering stats

2. Relax quality thresholds (modify `QualityFilter` in code)

3. Manually combine sub-clusters

### Issue: High Noise Percentage

**Symptoms**: >30% of images in `noise/` folder

**Causes**:
- `min_cluster_size` too high
- Highly diverse dataset
- Quality filtering too aggressive

**Solutions**:
1. Lower `min_cluster_size`:
   ```bash
   --min-cluster-size 5
   ```

2. Review noise samples - may contain useful data

---

## Best Practices

### 1. Start with Default Parameters

```bash
python3 character_clustering.py INPUT --output-dir OUTPUT
```

Review results before adjusting.

### 2. Cluster All Episodes Together

Better clustering with more data:
```bash
# All episodes → Single clustering run
```

### 3. Verify Quality Filtering

Check `quality_report.json`:
- Are good images being filtered?
- Are bad images passing through?

### 4. Use Visualization

`cluster_visualization.png` reveals:
- Cluster quality
- Outliers
- Optimal parameter settings

### 5. Iterative Refinement

1. Run with defaults
2. Review results
3. Adjust `min_cluster_size`
4. Re-run if needed

---

## Technical Details

### CLIP Models

| Model | Params | Embedding Size | Speed | Accuracy |
|-------|--------|----------------|-------|----------|
| ViT-B/32 | 151M | 512 | Fast | Good |
| ViT-B/16 | 149M | 512 | Medium | Better |
| ViT-L/14 | 428M | 768 | Slow | Best |

**Recommendation**: `ViT-B/32` for initial runs, `ViT-L/14` for final production.

### HDBSCAN Parameters

- **`min_cluster_size`**: Minimum points to form a cluster
  - Higher → Fewer, larger clusters
  - Lower → More, smaller clusters

- **`min_samples`**: Core point density
  - Fixed at 5 (good default)

- **`metric`**: Distance measure
  - Uses Euclidean on normalized embeddings

### Quality Metrics

**Blur Detection**:
- Laplacian variance: `∇²I`
- Threshold: 100 (empirically determined)

**Alpha Coverage**:
- Percentage of non-transparent pixels
- Threshold: 5% (filters empty/edge cases)

---

## Future Enhancements

Planned features:

1. **Temporal Consistency**
   - Group frames by scene/sequence
   - Reduce redundant similar frames

2. **Pose-Based Sub-Clustering**
   - Separate frontal/side/back views
   - Useful for varied training data

3. **Manual Labeling Interface**
   - Web UI for cluster inspection
   - Easy renaming and merging

4. **Multi-Modal Clustering**
   - Combine CLIP with metadata (episode, timestamp)
   - More robust character identification

---

## Summary

Character clustering automates the tedious work of organizing anime frames:

✅ **Automatic** - No manual labeling
✅ **Fast** - 16K images in ~30 minutes (GPU)
✅ **Quality** - Built-in filtering
✅ **Flexible** - Adjustable parameters
✅ **LoRA-ready** - Direct integration with training pipeline

**Next steps**:
1. Run clustering on your dataset
2. Review and rename clusters
3. Proceed to LoRA training

For questions, refer to inline code documentation in `scripts/tools/character_clustering.py`.
