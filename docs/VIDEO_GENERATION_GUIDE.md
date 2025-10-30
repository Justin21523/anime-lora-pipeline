# Video Generation and Processing Guide

Complete guide for overnight batch processing, LoRA testing, frame interpolation, and video synthesis.

## Overview

This guide covers the comprehensive video generation pipeline designed for 12-hour overnight processing sessions. The pipeline includes:

1. **Batch LoRA Testing** - Generate test images with all trained LoRA models
2. **Frame Interpolation** - Create smooth 60fps animations from 30fps source
3. **Video Synthesis** - Compile frames into high-quality videos with audio

## Installation

### Dependencies

Install video generation dependencies:

```bash
# Install Python dependencies
conda run -n blip2-env pip install -r requirements_video.txt

# Install PyTorch with CUDA support (if not already installed)
conda run -n blip2-env pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Optional: Install RIFE for high-quality frame interpolation
conda run -n blip2-env pip install rife-ncnn-vulkan-python
```

### Verify Installation

```bash
# Test LoRA generator
conda run -n blip2-env python3 scripts/tools/batch_lora_generator.py --help

# Test frame interpolator
conda run -n blip2-env python3 scripts/tools/frame_interpolator.py --help

# Test video synthesizer
conda run -n blip2-env python3 scripts/tools/video_synthesizer.py --help
```

## Quick Start

### Option 1: Run Complete Overnight Pipeline

The easiest way to utilize overnight idle time:

```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora

# Make script executable
chmod +x scripts/batch/overnight_processing.sh

# Run overnight processing
bash scripts/batch/overnight_processing.sh
```

This will:
- Test all trained LoRA models with comprehensive prompts
- Interpolate frames for selected episodes (every 10th episode)
- Synthesize 60fps videos with audio
- Run for approximately 10-12 hours
- Utilize 80-90% GPU capacity
- Require ~100-150 GB storage

### Option 2: Individual Tools

Use tools separately for more control:

#### A. Batch LoRA Testing

```bash
# Test a single LoRA model
conda run -n blip2-env python3 scripts/tools/batch_lora_generator.py \
  /path/to/lora.safetensors \
  --prompts prompts/inazuma_eleven_characters.json \
  --output-dir outputs/lora_test \
  --variations 5 \
  --steps 28 \
  --seed 42
```

#### B. Frame Interpolation

```bash
# Interpolate frames from one episode
conda run -n blip2-env python3 scripts/tools/frame_interpolator.py \
  /path/to/episode_frames \
  --output-dir outputs/interpolated \
  --multiplier 2 \
  --pattern "*.jpg" \
  --format jpg \
  --jpeg-quality 95
```

#### C. Video Synthesis

```bash
# Create video from frames
conda run -n blip2-env python3 scripts/tools/video_synthesizer.py \
  /path/to/frames \
  --output outputs/video.mp4 \
  --fps 60 \
  --audio /path/to/audio.wav \
  --crf 18 \
  --preset medium
```

## Detailed Usage

### 1. Batch LoRA Testing

Generate comprehensive test images to evaluate LoRA model quality.

**Parameters:**
- `lora_path`: Path to LoRA weights (.safetensors)
- `--base-model`: Base Stable Diffusion model path
- `--prompts`: Prompt library file (.txt or .json)
- `--output-dir`: Output directory for generated images
- `--variations`: Number of variations per prompt (default: 1)
- `--steps`: Inference steps (default: 28)
- `--cfg`: Guidance scale (default: 7.5)
- `--seed`: Random seed for reproducibility

**Example with all options:**

```bash
conda run -n blip2-env python3 scripts/tools/batch_lora_generator.py \
  /path/to/endou_mamoru_v1.safetensors \
  --base-model /warehouse/models/anything-v4.5.safetensors \
  --prompts prompts/inazuma_eleven_characters.json \
  --output-dir outputs/endou_mamoru_test \
  --variations 5 \
  --steps 28 \
  --cfg 7.5 \
  --width 512 \
  --height 512 \
  --seed 42 \
  --negative "lowres, bad quality, blurry"
```

**Output:**
- `img_XXXXX_pYYY_vZZ.png` - Generated images
- `generation_metadata.json` - Complete generation parameters

**Prompt Library Format:**

JSON format:
```json
{
  "prompts": [
    "character name, description, style, quality",
    "another prompt here"
  ]
}
```

Text format (one prompt per line):
```
character name, description, style, quality
another prompt here
```

### 2. Frame Interpolation

Create smooth animations by generating intermediate frames.

**Parameters:**
- `input_dir`: Directory with sequential frames
- `--output-dir`: Output directory
- `--multiplier`: Frame rate multiplier (2x, 4x, 8x)
- `--pattern`: Input file pattern (default: *.jpg)
- `--format`: Output format (jpg/png)
- `--jpeg-quality`: JPEG quality if using jpg format
- `--model`: RIFE model version (default: rife-v4.6)
- `--no-rife`: Disable RIFE, use basic interpolation

**Example:**

```bash
# 2x interpolation (30fps -> 60fps)
conda run -n blip2-env python3 scripts/tools/frame_interpolator.py \
  cache/episode_001 \
  --output-dir outputs/episode_001_60fps \
  --multiplier 2 \
  --pattern "*.jpg" \
  --format jpg \
  --jpeg-quality 95
```

**RIFE vs Basic Interpolation:**
- RIFE: High-quality AI-based interpolation, slower, requires RIFE package
- Basic: Simple linear blending, faster, always available

**Storage Impact:**
- 2x multiplier: Doubles frame count and storage
- 4x multiplier: Quadruples frame count and storage
- Example: 1000 frames @ 2MB each = 2GB raw, 4GB with 2x interpolation

### 3. Video Synthesis

Compile frame sequences into video files with optional audio.

**Parameters:**
- `input_dir`: Directory with frames
- `--output`: Output video path
- `--pattern`: Frame filename pattern
- `--fps`: Frames per second (default: 30)
- `--audio`: Optional audio file
- `--codec`: Video codec (default: libx264)
- `--crf`: Constant Rate Factor, 0-51 (default: 18, lower=better)
- `--preset`: Encoding speed (ultrafast/fast/medium/slow/veryslow)
- `--resolution`: Output resolution (e.g., 1920x1080)

**Single Video Example:**

```bash
conda run -n blip2-env python3 scripts/tools/video_synthesizer.py \
  outputs/episode_001_60fps \
  --output videos/episode_001.mp4 \
  --fps 60 \
  --audio audio/episode_001.wav \
  --crf 18 \
  --preset medium \
  --resolution 1920x1080
```

**Batch Processing Example:**

```bash
# Process multiple frame directories
conda run -n blip2-env python3 scripts/tools/video_synthesizer.py \
  outputs/interpolated_frames \
  --batch \
  --output-dir videos/all_episodes \
  --audio-dir audio/separated/vocals \
  --fps 60 \
  --crf 18
```

**Quality Settings:**

CRF (Constant Rate Factor):
- 0-17: Visually lossless (very large files)
- 18-23: High quality (recommended for archival)
- 24-28: Good quality (balanced)
- 29-35: Medium quality (streaming)
- 36-51: Low quality (not recommended)

Preset (speed vs compression):
- ultrafast: Fastest encoding, largest files
- fast: Quick encoding, large files
- medium: Balanced (recommended)
- slow: Slow encoding, smaller files
- veryslow: Very slow, smallest files (diminishing returns)

## Workflow Examples

### Workflow 1: Complete LoRA Testing

Test all trained LoRAs comprehensively:

```bash
#!/bin/bash
LORA_DIR="/warehouse/models/lora/character_loras/inazuma-eleven"
OUTPUT_DIR="/warehouse/outputs/lora_testing"

for lora in "$LORA_DIR"/*.safetensors; do
    name=$(basename "$lora" .safetensors)

    conda run -n blip2-env python3 scripts/tools/batch_lora_generator.py \
        "$lora" \
        --prompts prompts/inazuma_eleven_characters.json \
        --output-dir "$OUTPUT_DIR/$name" \
        --variations 5 \
        --steps 28 \
        --seed 42
done
```

### Workflow 2: Episode to 60fps Video

Complete pipeline for one episode:

```bash
#!/bin/bash
EPISODE="episode_042"
FRAMES_DIR="cache/extracted_frames/$EPISODE"
AUDIO_FILE="cache/audio/${EPISODE}.wav"
OUTPUT_DIR="outputs/final_videos"

# 1. Interpolate frames
conda run -n blip2-env python3 scripts/tools/frame_interpolator.py \
    "$FRAMES_DIR" \
    --output-dir "temp/interpolated_$EPISODE" \
    --multiplier 2

# 2. Synthesize video
conda run -n blip2-env python3 scripts/tools/video_synthesizer.py \
    "temp/interpolated_$EPISODE" \
    --output "$OUTPUT_DIR/${EPISODE}_60fps.mp4" \
    --fps 60 \
    --audio "$AUDIO_FILE" \
    --crf 18 \
    --preset medium
```

### Workflow 3: Comparison Grid

Create side-by-side comparison of different versions:

```bash
# Create comparison of original vs interpolated
conda run -n blip2-env python3 scripts/tools/video_synthesizer.py \
    --create-grid \
    --videos original.mp4 interpolated.mp4 \
    --output comparison_2x1.mp4 \
    --layout 2x1
```

## Monitoring Progress

### Check Running Processes

```bash
# View all GPU processes
nvidia-smi

# View LoRA generation processes
ps aux | grep batch_lora_generator

# View interpolation processes
ps aux | grep frame_interpolator

# View video synthesis processes
ps aux | grep video_synthesizer
```

### Monitor Logs

```bash
# Tail overnight processing logs
tail -f /warehouse/outputs/overnight_*/logs/*.log

# Check specific task
tail -f /warehouse/outputs/overnight_*/logs/lora_endou_mamoru.log
```

### Check Disk Usage

```bash
# Monitor output directory size
watch -n 60 du -sh /warehouse/outputs/overnight_*

# Check available space
df -h /mnt/c
```

## Performance Optimization

### GPU Utilization

For maximum overnight throughput:

```bash
# In overnight_processing.sh, adjust:
MAX_CONCURRENT_LORA_JOBS=2    # Increase if GPU underutilized
MAX_CONCURRENT_VIDEO_JOBS=4   # CPU-bound, can be higher
```

### Memory Management

If encountering OOM errors:

```python
# In batch_lora_generator.py, reduce batch size:
num_images_per_prompt=1  # Instead of 5

# Or reduce resolution:
--width 448 --height 448  # Instead of 512x512
```

### Storage Management

Manage disk space during long runs:

```bash
# Clean up intermediate files periodically
find /warehouse/cache/temp -name "*.jpg" -mtime +1 -delete

# Compress completed outputs
tar -czf lora_tests.tar.gz /warehouse/outputs/lora_testing/
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce concurrent jobs
MAX_CONCURRENT_LORA_JOBS=1

# Use smaller image dimensions
--width 448 --height 448

# Enable CPU offloading in script
pipe.enable_model_cpu_offload()
```

### RIFE Not Available

```bash
# Install RIFE
conda run -n blip2-env pip install rife-ncnn-vulkan-python

# Or use basic interpolation
python3 scripts/tools/frame_interpolator.py \
    input_dir \
    --output-dir output \
    --no-rife
```

### FFmpeg Errors

```bash
# Check ffmpeg installation
ffmpeg -version

# Install if missing
sudo apt update
sudo apt install ffmpeg

# Test codec support
ffmpeg -codecs | grep h264
```

### Prompt File Not Found

```bash
# Create your own prompt file
cat > custom_prompts.txt <<EOF
character name, action pose, high quality
another test prompt here
EOF

# Use with LoRA generator
--prompts custom_prompts.txt
```

## Best Practices

### 1. Start Small, Scale Up

```bash
# Test with 1 LoRA first
python3 scripts/tools/batch_lora_generator.py \
    single_lora.safetensors \
    --variations 2 \
    --prompt "test prompt"

# Then scale to full batch
bash scripts/batch/overnight_processing.sh
```

### 2. Use Seeds for Reproducibility

```bash
# Always use consistent seed for testing
--seed 42
```

### 3. Quality vs Storage Tradeoff

```bash
# High quality (large files)
--crf 18 --preset slow --jpeg-quality 95

# Balanced (recommended)
--crf 23 --preset medium --jpeg-quality 90

# Storage-efficient
--crf 28 --preset fast --jpeg-quality 85
```

### 4. Backup Important Results

```bash
# Backup generated content
rsync -av /warehouse/outputs/overnight_* /backup/location/
```

## Output Structure

After overnight processing completes:

```
outputs/overnight_processing_YYYYMMDD_HHMMSS/
├── config.json                         # Processing configuration
├── summary.txt                         # Completion summary
├── lora_testing/                       # LoRA test images
│   ├── endou_mamoru_v1/
│   │   ├── img_00000_p000_v00.png
│   │   ├── img_00001_p000_v01.png
│   │   └── generation_metadata.json
│   └── gouenji_shuuya_v1/
│       └── ...
├── interpolated_frames/                # 60fps frame sequences
│   ├── episode_001/
│   │   ├── frame_000000.jpg
│   │   └── interpolation_metadata.json
│   └── episode_011/
│       └── ...
├── synthesized_videos/                 # Final videos
│   ├── episode_001_60fps.mp4
│   └── episode_011_60fps.mp4
└── logs/                               # Processing logs
    ├── lora_endou_mamoru_v1.log
    ├── interpolation_episode_001.log
    └── video_episode_001.log
```

## Next Steps

After overnight processing:

1. **Review LoRA Quality**
   ```bash
   ls outputs/*/lora_testing/*/img_*.png | xargs feh
   ```

2. **Watch Generated Videos**
   ```bash
   ffplay outputs/*/synthesized_videos/*.mp4
   ```

3. **Analyze Results**
   ```bash
   python3 scripts/tools/analyze_lora_quality.py outputs/*/lora_testing
   ```

4. **Select Best Models**
   - Use highest quality LoRAs for production
   - Archive or retrain underperforming models

## Additional Resources

- [RIFE GitHub](https://github.com/nihui/rife-ncnn-vulkan)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LoRA Training Guide](../USAGE_GUIDE.md)

## Support

For issues or questions:
- Check logs in `outputs/overnight_*/logs/`
- Review error messages in console output
- Verify dependencies with `pip list`
- Check GPU status with `nvidia-smi`
