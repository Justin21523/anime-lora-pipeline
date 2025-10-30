# Anime Video Processing Pipeline - Setup and Usage Guide

## Overview

This professional-grade pipeline processes anime video frames through multiple AI-powered stages:

1. **Stage 1**: Base Segmentation (Mask2Former)
2. **Stage 2a**: Character Refinement (U²-Net/MODNet)
3. **Stage 2b**: Effect Separation (CLIPSeg/Grounded-SAM)
4. **Stage 2c**: Background Inpainting (LaMa)
5. **Stage 3**: Temporal Consistency (XMem/DeAOT)
6. **Stage 4**: Intelligent Annotation (CLIP/BLIP-2/DINOv2)

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ GPU memory recommended
- Ubuntu 20.04+ or WSL2

### Step 1: Create Conda Environment

```bash
conda create -n anime-pipeline python=3.10
conda activate anime-pipeline
```

### Step 2: Install PyTorch

```bash
# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Core Dependencies

```bash
pip install opencv-python pillow numpy tqdm
pip install transformers accelerate
pip install timm einops
```

### Step 4: Install Stage-Specific Models

#### Stage 1: Mask2Former

```bash
# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Mask2Former
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
python setup.py build develop
cd ..
```

#### Stage 2a: U²-Net and MODNet

```bash
# U²-Net
pip install git+https://github.com/xuebinqin/U-2-Net.git

# MODNet
pip install git+https://github.com/ZHKKKe/MODNet.git
```

#### Stage 2b: CLIPSeg and Grounded-SAM

```bash
# CLIPSeg
pip install git+https://github.com/timojl/clipseg.git

# Grounded-SAM
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
cd Grounded-Segment-Anything
pip install -e segment_anything
pip install -e GroundingDINO
cd ..
```

#### Stage 2c: LaMa

```bash
pip install lama-cleaner
```

#### Stage 3: XMem

```bash
git clone https://github.com/hkchengrex/XMem
cd XMem
pip install -e .
cd ..
```

#### Stage 4: Already installed (transformers includes CLIP, BLIP-2)

```bash
pip install sentence-transformers  # For DINOv2 if needed
```

## Model Weights Download

### Mask2Former (COCO-trained)

```bash
mkdir -p models/mask2former
cd models/mask2former
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final.pth
cd ../..
```

### U²-Net

```bash
mkdir -p models/u2net
cd models/u2net
wget https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth
cd ../..
```

### MODNet

```bash
mkdir -p models/modnet
cd models/modnet
wget https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_photographic_portrait_matting.ckpt
cd ../..
```

## Usage

### Quick Start (Full Pipeline)

Process video frames through all stages:

```bash
python scripts/pipeline/orchestrator.py \
    /path/to/input/frames \
    -o /path/to/output \
    --device cuda
```

### Run Specific Stages

#### Stage 1 Only (Base Segmentation)

```bash
python scripts/pipeline/stage1_segmentation.py \
    /path/to/input/frames \
    -o /path/to/output/stage1 \
    --weights models/mask2former/model_final.pth \
    --device cuda
```

#### Stage 2a Only (Character Refinement)

```bash
python scripts/pipeline/stage2a_character_refine.py \
    /path/to/input/frames \
    -o /path/to/output/stage2a \
    --model u2net \
    --weights models/u2net/u2net.pth \
    --coarse-masks /path/to/stage1/masks \
    --device cuda
```

### Advanced Configuration

Create a configuration file `pipeline_config.json`:

```json
{
    "device": "cuda",
    "batch_size": 8,
    "num_workers": 4,
    
    "stage1": {
        "enabled": true,
        "config_file": "configs/mask2former_config.yaml",
        "weights_file": "models/mask2former/model_final.pth",
        "confidence_threshold": 0.5
    },
    
    "stage2a": {
        "enabled": true,
        "model_type": "u2net",
        "weights_path": "models/u2net/u2net.pth"
    },
    
    "stage2b": {
        "enabled": true,
        "use_clipseg": true,
        "effect_prompts": [
            "glowing energy effect",
            "fire and flames",
            "light beam"
        ]
    },
    
    "stage2c": {
        "enabled": true,
        "method": "lama"
    },
    
    "stage3": {
        "enabled": false
    },
    
    "stage4": {
        "enabled": true,
        "use_clip": true,
        "use_blip2": true
    }
}
```

Run with config:

```bash
python scripts/pipeline/orchestrator.py \
    /path/to/input/frames \
    -o /path/to/output \
    --config pipeline_config.json
```

### Partial Pipeline Execution

Run only specific stages:

```bash
# Run stages 1-2 only
python scripts/pipeline/orchestrator.py \
    /path/to/input \
    -o /path/to/output \
    --start-stage 1 \
    --end-stage 2

# Run stage 4 only (annotation on existing results)
python scripts/pipeline/orchestrator.py \
    /path/to/stage2_output \
    -o /path/to/output \
    --start-stage 4 \
    --end-stage 4
```

## Output Structure

```
output/
├── stage1_segmentation/
│   └── masks/
│       ├── frame_0001_mask.png
│       ├── frame_0001_semantic.png
│       └── ...
├── stage2a_character_refine/
│   ├── refined_masks/
│   │   └── frame_0001_refined_mask.png
│   └── characters_rgba/
│       └── frame_0001_character.png
├── stage2b_effect_separation/
│   └── effects/
│       └── frame_0001_effect.png
├── stage2c_background_inpaint/
│   └── backgrounds/
│       └── frame_0001_background.jpg
├── stage3_temporal_consistency/
│   └── smoothed/
├── stage4_annotation/
│   ├── annotations.json
│   └── embeddings.npy
└── pipeline_report.json
```

## Performance Optimization

### GPU Utilization Tips

1. **Batch Size**: Increase if you have more GPU memory
   ```bash
   python scripts/pipeline/orchestrator.py ... --batch-size 16
   ```

2. **Mixed Precision**: Enable FP16 for faster inference
   - Modify model loading to use `torch.cuda.amp`

3. **Model Quantization**: Use INT8 for deployment
   ```python
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

### Multi-GPU Support

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/pipeline/orchestrator.py ...
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size: `--batch-size 4`
- Process fewer frames at once
- Use CPU for heavy models: `--device cpu`

### Slow Processing

- Check GPU utilization: `nvidia-smi`
- Increase `num_workers` for data loading
- Use SSD for frame storage (not HDD)

### Model Loading Errors

- Ensure all dependencies are installed
- Download correct model weights
- Check CUDA compatibility

## Example Workflows

### Workflow 1: Extract High-Quality Character Layers

```bash
# Stage 1: Coarse segmentation
python scripts/pipeline/stage1_segmentation.py \
    input_frames/ -o output/stage1

# Stage 2a: Refine character boundaries
python scripts/pipeline/stage2a_character_refine.py \
    input_frames/ -o output/stage2a \
    --coarse-masks output/stage1/masks \
    --model u2net

# Result: High-quality RGBA character layers in output/stage2a/characters_rgba/
```

### Workflow 2: Full Layered Decomposition

```bash
# Run complete pipeline
python scripts/pipeline/orchestrator.py \
    input_frames/ -o output/full_pipeline \
    --config full_config.json

# Result: 
# - Characters in output/stage2a/characters_rgba/
# - Effects in output/stage2b/effects/
# - Backgrounds in output/stage2c/backgrounds/
# - Annotations in output/stage4/annotations.json
```

## Next Steps

- Fine-tune models on your specific anime dataset
- Implement video temporal consistency (Stage 3)
- Create custom effect detection prompts (Stage 2b)
- Build training datasets from layered outputs

## Support and Contributing

- Report issues: GitHub Issues
- Documentation: See `docs/ADVANCED_VIDEO_PIPELINE.md`
- Contributing: Pull requests welcome

## References

- Mask2Former: https://github.com/facebookresearch/Mask2Former
- U²-Net: https://github.com/xuebinqin/U-2-Net
- MODNet: https://github.com/ZHKKKe/MODNet
- XMem: https://github.com/hkchengrex/XMem
- CLIP: https://github.com/openai/CLIP
