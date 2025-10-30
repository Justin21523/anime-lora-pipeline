# Layered Segmentation Guide

Comprehensive guide for separating anime frames into character layers, background layers, and effect layers.

## Overview

Layered segmentation is a crucial step in the video processing pipeline. It separates each frame into multiple semantic layers that can be independently processed, modified, or used for training different generation models.

**Based on**: Video Processing and Generation Analysis research document

### Why Layered Segmentation?

Traditional frame extraction gives you static images. Layered segmentation transforms them into **structured, reusable assets**:

- **Character Layer**: Clean character with alpha channel → LoRA training
- **Background Layer**: Inpainted scene without characters → Background generation
- **Effect Layer** (optional): Special effects like transformations, auras → Effect generation

### Applications

1. **Character-Specific LoRA Training**: Train on isolated characters without background interference
2. **Background Generation**: Learn scene composition independently
3. **Motion Transfer**: Swap characters between scenes while preserving backgrounds
4. **Effect Generation**: Extract and apply special effects to different characters
5. **Style Transfer**: Apply different art styles to layers separately

---

## Installation

### Dependencies

```bash
# Install segmentation dependencies
conda run -n blip2-env pip install -r requirements_segmentation.txt

# Core requirements:
# - rembg[gpu] >= 2.0.0          # Character segmentation
# - onnxruntime-gpu >= 1.23.0    # GPU acceleration
# - opencv-python >= 4.8.0       # Image processing
```

### Verify Installation

```bash
conda run -n blip2-env python3 scripts/tools/layered_segmentation.py --help
```

---

## Quick Start

### Basic Usage

Process frames from one episode:

```bash
conda run -n blip2-env python3 scripts/tools/layered_segmentation.py \
  /path/to/episode_001
```

This will create `episode_001_layers` directory with:
- `character/` - Character layers with alpha channel (PNG)
- `background/` - Inpainted backgrounds (JPG)
- `masks/` - Binary segmentation masks (PNG)

### Custom Output Directory

```bash
python3 scripts/tools/layered_segmentation.py \
  /path/to/episode_001 \
  --output-dir /path/to/output_layers
```

---

## Model Options

### Segmentation Models

The tool supports multiple models via **rembg**:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **u2net** (default) | Fast | Good | General anime, recommended |
| **u2netp** | Very Fast | Moderate | Quick testing |
| **isnet-anime** | Moderate | Excellent | Best for 2D anime |
| **isnet-general-use** | Moderate | Good | Mixed content |

**Example with different models:**

```bash
# Fast processing (default)
python3 scripts/tools/layered_segmentation.py episode_001 --model u2net

# Best quality for anime
python3 scripts/tools/layered_segmentation.py episode_001 --model isnet-anime

# Fastest for testing
python3 scripts/tools/layered_segmentation.py episode_001 --model u2netp
```

### Inpainting Methods

Two methods for background reconstruction:

| Method | Quality | Speed | Requirements |
|--------|---------|-------|--------------|
| **telea** (default) | Good | Fast | OpenCV only |
| **lama** | Excellent | Slow | simple-lama-inpainting |

**Example:**

```bash
# Fast traditional method (default)
python3 scripts/tools/layered_segmentation.py episode_001 --inpaint telea

# High-quality deep learning (requires additional package)
pip install simple-lama-inpainting
python3 scripts/tools/layered_segmentation.py episode_001 --inpaint lama
```

---

## Advanced Usage

### Processing Multiple Episodes

Process all episodes in a directory:

```bash
for ep in /path/to/episodes/episode_*; do
    python3 scripts/tools/layered_segmentation.py "$ep" \
        --output-dir "/path/to/layers/$(basename $ep)_layers"
done
```

### Batch Processing Script

Create a simple batch script:

```bash
#!/bin/bash
EPISODES_DIR="/path/to/extracted_frames"
OUTPUT_DIR="/path/to/layered_output"

for episode in "$EPISODES_DIR"/episode_*; do
    name=$(basename "$episode")
    echo "Processing $name..."

    python3 scripts/tools/layered_segmentation.py \
        "$episode" \
        --output-dir "$OUTPUT_DIR/${name}_layers" \
        --model isnet-anime \
        --inpaint telea
done

echo "All episodes processed!"
```

### GPU vs CPU

```bash
# Use GPU (default, much faster)
python3 scripts/tools/layered_segmentation.py episode_001 --device cuda

# Force CPU (slower, for systems without GPU)
python3 scripts/tools/layered_segmentation.py episode_001 --device cpu
```

---

## Output Structure

After processing `episode_001`, you'll get:

```
episode_001_layers/
├── character/                    # Character layers (RGBA PNG)
│   ├── scene0000_frame000000_t1.00s_character.png
│   ├── scene0000_frame000001_t5.00s_character.png
│   └── ...
├── background/                   # Inpainted backgrounds (RGB JPG)
│   ├── scene0000_frame000000_t1.00s_background.jpg
│   ├── scene0000_frame000001_t5.00s_background.jpg
│   └── ...
├── masks/                        # Binary masks (Grayscale PNG)
│   ├── scene0000_frame000000_t1.00s_mask.png
│   ├── scene0000_frame000001_t5.00s_mask.png
│   └── ...
└── segmentation_results.json     # Processing metadata
```

### File Naming Convention

Format: `{original_name}_{layer_type}.{ext}`

Example:
- `scene0042_frame002156_t847.25s_character.png`
  - Original: scene 42, frame 2156, at 847.25 seconds
  - Layer: character with alpha channel

- `scene0042_frame002156_t847.25s_background.jpg`
  - Same frame, inpainted background

### Metadata JSON

`segmentation_results.json` contains:

```json
{
  "timestamp": "2025-10-27T15:55:30",
  "input_dir": "/path/to/episode_001",
  "output_dir": "/path/to/episode_001_layers",
  "total_frames": 1236,
  "config": {
    "seg_model": "u2net",
    "inpaint_method": "telea",
    "device": "cuda"
  },
  "processed_frames": [
    {
      "input": "scene0000_frame000000_t1.00s.jpg",
      "outputs": {
        "character": "scene0000_frame000000_t1.00s_character.png",
        "background": "scene0000_frame000000_t1.00s_background.jpg",
        "mask": "scene0000_frame000000_t1.00s_mask.png"
      }
    }
  ]
}
```

---

## Performance & Optimization

### Processing Speed

Typical performance on modern hardware:

| Model | GPU (RTX 3080) | CPU (16-core) |
|-------|----------------|---------------|
| u2net | ~2-3 sec/frame | ~15-20 sec/frame |
| u2netp | ~1-2 sec/frame | ~10-15 sec/frame |
| isnet-anime | ~3-4 sec/frame | ~20-25 sec/frame |

**For 1236 frames (one episode)**:
- u2net + GPU: ~40-60 minutes
- u2net + CPU: ~5-7 hours

### Storage Requirements

Per episode (1236 frames):

- **Character layers** (PNG, RGBA): ~200-400 MB
- **Background layers** (JPG 95%): ~40-80 MB
- **Masks** (PNG, 1-bit): ~2-5 MB
- **Total**: ~250-500 MB per episode

For 13 episodes: **3.25-6.5 GB**

### Optimization Tips

1. **Use u2net for balance** (default, recommended)
2. **GPU is essential** for reasonable processing time
3. **Process in parallel**:
   ```bash
   # Process 4 episodes simultaneously
   for i in {1..4}; do
       python3 layered_segmentation.py episode_00$i &
   done
   wait
   ```

4. **Monitor GPU memory**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

## Integration with Training Pipeline

### Step 1: Extract and Segment Frames

```bash
# 1. Extract frames from videos
python3 scripts/tools/universal_frame_extractor.py /path/to/videos

# 2. Segment frames into layers
python3 scripts/tools/layered_segmentation.py \
    /path/to/extracted_frames/episode_001
```

### Step 2: Use Character Layers for LoRA

```bash
# Character layers are now ready for LoRA training
# They have clean edges and no background interference

cp episode_001_layers/character/*.png \
   /path/to/lora_training_data/endou_mamoru/
```

### Step 3: Background Generation Training

```bash
# Use background layers for scene generation models
cp episode_001_layers/background/*.jpg \
   /path/to/background_training_data/
```

### Step 4: Advanced: Combined Generation

With separated layers, you can:
1. Generate character with LoRA
2. Generate background with scene model
3. Composite them with proper blending

---

## Technical Details

### How It Works

The pipeline consists of three main steps:

#### 1. Character Segmentation

Uses **rembg** (U²-Net architecture) to detect characters:

```
Input Frame → U²-Net Model → Binary Mask
            ↓
    Alpha Matting → Character Layer (RGBA)
```

- Detects anime character boundaries
- Generates precise alpha channel
- Preserves fine details (hair, clothing edges)

#### 2. Background Inpainting

Fills the region where character was removed:

```
Input Frame + Mask → Inpainting → Clean Background
```

**Telea Algorithm** (default):
- Fast traditional method
- Good for simple backgrounds
- Based on Navier-Stokes equations

**LaMa** (optional):
- Deep learning method
- Better for complex scenes
- Handles large missing regions

#### 3. Post-Processing

- **Edge Refinement**: Morphological operations to smooth edges
- **Mask Cleanup**: Remove noise and small artifacts
- **Alpha Blending**: Smooth character-background transition

### Model Comparison

**U²-Net Architecture**:
- Nested U-structure
- Multi-scale feature extraction
- Excellent for salient object detection

**ISNet**:
- Improved segmentation network
- Better boundary accuracy
- Specialized anime version available

---

## Troubleshooting

### Issue: Segmentation Quality Poor

**Symptoms**: Characters not fully separated, holes in masks

**Solutions**:
1. Try different model:
   ```bash
   --model isnet-anime  # Better for anime
   ```

2. Check input quality:
   - Are frames blurry?
   - Are characters too small?
   - Is lighting too dark?

3. Refine edges (already enabled by default)

### Issue: Background Inpainting Has Artifacts

**Symptoms**: Visible seams, unnatural textures in background

**Solutions**:
1. Use LaMa for better quality:
   ```bash
   pip install simple-lama-inpainting
   --inpaint lama
   ```

2. Acceptable for training data (models learn to ignore small artifacts)

### Issue: Out of Memory

**Symptoms**: CUDA out of memory errors

**Solutions**:
1. Process fewer frames at once
2. Use smaller model:
   ```bash
   --model u2netp  # Lighter version
   ```
3. Use CPU (slower):
   ```bash
   --device cpu
   ```

### Issue: Very Slow Processing

**Solutions**:
1. Ensure using GPU:
   ```bash
   nvidia-smi  # Check GPU is detected
   --device cuda  # Explicitly use GPU
   ```

2. Use faster model:
   ```bash
   --model u2netp
   ```

3. Process in parallel (multiple episodes)

---

## Best Practices

### 1. Always Test First

Process a small sample before running on all data:

```bash
# Create test set
mkdir test_frames
cp episode_001/*.jpg test_frames/ | head -50

# Test segmentation
python3 layered_segmentation.py test_frames --model isnet-anime
```

Review quality, then process full dataset.

### 2. Model Selection

- **For speed**: u2net or u2netp
- **For quality**: isnet-anime
- **For mixed content**: isnet-general-use

### 3. Storage Management

```bash
# Compress character layers if needed (lossy)
mogrify -quality 95 -format jpg character/*.png

# Or keep PNG for lossless (recommended for training)
```

### 4. Quality Check

Periodically review outputs:

```bash
# Random sample check
ls character/*.png | shuf | head -10 | xargs feh
```

### 5. Backup Original Frames

Keep original frames separate:
- Layered data can always be regenerated
- Original frames are irreplaceable

---

## Advanced: Character Clustering

After segmentation, you can cluster characters:

**Step 1: Extract character embeddings**
```python
# Use CLIP to get character features
from PIL import Image
import clip

model, preprocess = clip.load("ViT-B/32")

for char_img in character_layers:
    image = preprocess(Image.open(char_img)).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image)
    # Save features for clustering
```

**Step 2: Cluster by character**
```python
# Use HDBSCAN or K-means to group similar characters
from sklearn.cluster import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(features)

# Result: "These 500 frames are Endou Mamoru"
```

This enables:
- Auto-organizing frames by character
- Character-specific training sets
- Quality filtering (remove minor characters)

---

## Future Enhancements

Planned additions:

1. **Effect Layer Extraction**
   - Detect and separate special effects (auras, energy, transformations)
   - Use CLIPSeg with text prompts ("glowing aura", "fire effect")

2. **Pose Estimation**
   - Extract character skeleton from character layer
   - Enable pose-to-image generation

3. **Face Detection & Alignment**
   - Detect and crop character faces
   - Align faces for consistent training

4. **Video Temporal Consistency**
   - Use optical flow to maintain mask consistency across frames
   - Reduce flickering in segmentation

5. **Batch Processing Dashboard**
   - Web interface for monitoring progress
   - Preview results in real-time
   - Adjust parameters on-the-fly

---

## References

- **Research Foundation**: `ChatGPT-影片處理與生成分析.md` (Video Processing and Generation Analysis)
- **Rembg**: https://github.com/danielgatis/rembg
- **U²-Net**: https://arxiv.org/abs/2005.09007
- **ISNet**: https://github.com/xuebinqin/DIS

---

## Summary

Layered segmentation is a **game-changer** for anime frame processing:

✅ **Separates frames into reusable layers**
✅ **Enables character-specific LoRA training**
✅ **Provides clean backgrounds for scene generation**
✅ **Fast with GPU acceleration**
✅ **Easy to integrate into existing pipelines**

**Next Steps**:
1. Process your frames: `python3 layered_segmentation.py episode_001`
2. Review outputs in character/background folders
3. Use layers for targeted training tasks

For questions or issues, refer to the Troubleshooting section or check the inline code documentation.
