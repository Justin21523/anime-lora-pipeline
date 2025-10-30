# LoRA Testing Guide

This guide explains how to test and evaluate your trained LoRA models using the automated testing script.

## Overview

The `test_lora_checkpoints.py` script provides a complete workflow for:

1. **Automatic Checkpoint Discovery**: Finds all `.safetensors` files in your LoRA output directory
2. **Test Image Generation**: Generates test images using each checkpoint with standard prompts
3. **Quality Evaluation**: Uses CLIP scores and consistency metrics to evaluate each checkpoint
4. **Comparison**: Compares all checkpoints side-by-side with visualizations

## Quick Start

### Test Battle LoRA Checkpoints

```bash
python3 scripts/evaluation/test_lora_checkpoints.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/outputs/lora_training/inazuma_battle_sd15 \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors \
  --output-dir outputs/lora_testing/battle_sd15_test \
  --device cuda
```

### Test Diverse LoRA Checkpoints

```bash
python3 scripts/evaluation/test_lora_checkpoints.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/outputs/lora_training/inazuma_diverse_sd15 \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors \
  --output-dir outputs/lora_testing/diverse_sd15_test \
  --device cuda
```

## Parameters

### Required
- `lora_dir`: Directory containing your trained LoRA `.safetensors` files
- `--base-model`: Path to your Stable Diffusion base model
- `--output-dir`: Where to save test results

### Optional Generation Parameters
- `--num-variations`: Number of images per prompt (default: 4)
- `--steps`: Inference steps (default: 25)
- `--cfg-scale`: Guidance scale (default: 7.5)
- `--width`: Image width (default: 512)
- `--height`: Image height (default: 512)
- `--seed`: Base seed for reproducibility (default: 42)

### Optional Workflow Control
- `--prompts-file`: Use custom prompts from text file (one per line)
- `--skip-generation`: Only run evaluation on existing images
- `--skip-evaluation`: Only generate images, skip quality metrics
- `--device`: Device to use (cuda/cpu)

## Custom Prompts

Create a custom prompts file (`prompts.txt`):

```
inazuma eleven character, soccer player, running with ball
inazuma eleven character, goalkeeper diving
inazuma eleven character, school uniform portrait
```

Then use it:

```bash
python3 scripts/evaluation/test_lora_checkpoints.py \
  your_lora_dir \
  --base-model your_model.safetensors \
  --output-dir test_output \
  --prompts-file prompts.txt
```

## Output Structure

所有測試結果會儲存在專案的 `outputs/` 目錄下:

```
inazuma-eleven-lora/
└── outputs/
    └── lora_testing/
        ├── battle_sd15_test/
        │   ├── inazuma_battle_v1-000003/      # Epoch 3 results
        │   │   ├── img_00000_p000_v00.png     # Generated images
        │   │   ├── img_00000_p000_v01.png
        │   │   ├── ...
        │   │   ├── generation_metadata.json    # Generation parameters
        │   │   └── quality_evaluation.json     # Quality metrics
        │   ├── inazuma_battle_v1-000006/       # Epoch 6 results
        │   ├── ...
        │   ├── all_checkpoints_results.json
        │   ├── checkpoint_comparison.json
        │   └── checkpoint_comparison.png
        └── diverse_sd15_test/
            └── (same structure)
```

## Understanding Results

### CLIP Score (Prompt Adherence)
- **Excellent (≥30)**: Strong prompt following
- **Good (25-30)**: Adequate prompt following
- **Needs Improvement (<25)**: Weak prompt following

### Character Consistency
- **Excellent (≥85%)**: Very consistent character appearance
- **Good (75-85%)**: Fairly consistent
- **Variable (<75%)**: Consider retraining

### Interpreting Comparison

The comparison will show you:
- Which epoch performs best for each metric
- How much variation exists between checkpoints
- Whether additional training improved or degraded quality

**Common Patterns:**
- **Improving trend**: Each epoch better than the last → Training working well
- **Peak then decline**: Best at middle epoch → Possible overfitting
- **Stable performance**: Similar across epochs → Well-balanced dataset

## Tips

1. **Start with default settings** to get baseline results
2. **Use 4-8 variations per prompt** for reliable consistency metrics
3. **Test with diverse prompts** covering different scenarios
4. **Compare at least 3 checkpoints** to see training progression
5. **Re-test top 2 checkpoints** with more prompts/variations for final selection

## Advanced Usage

### Testing Specific Checkpoints Only

Move checkpoints you want to test to a temporary directory:

```bash
mkdir temp_test
cp lora_dir/inazuma_v1-000009.safetensors temp_test/
cp lora_dir/inazuma_v1.safetensors temp_test/

python3 scripts/evaluation/test_lora_checkpoints.py temp_test --base-model ... --output-dir ...
```

### High-Quality Evaluation

For final model selection, use more images:

```bash
python3 scripts/evaluation/test_lora_checkpoints.py \
  your_lora_dir \
  --base-model your_model.safetensors \
  --output-dir test_output_hq \
  --num-variations 8 \
  --steps 30 \
  --prompts-file extensive_prompts.txt
```

### Re-run Evaluation Only

If you already generated images:

```bash
python3 scripts/evaluation/test_lora_checkpoints.py \
  your_lora_dir \
  --base-model your_model.safetensors \
  --output-dir existing_test_output \
  --skip-generation
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or image dimensions:
```bash
--width 448 --height 448
```

Or test one checkpoint at a time.

### CLIP Not Installed

Install required packages:
```bash
pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git
```

### Diffusers Issues

Ensure you have the latest diffusers:
```bash
pip install --upgrade diffusers transformers accelerate
```

## Integration with Training Pipeline

You can add testing to your training workflow:

```bash
# 1. Train LoRAs
bash scripts/batch/auto_train_sequence.sh

# 2. Test Battle LoRA
python3 scripts/evaluation/test_lora_checkpoints.py \
  outputs/lora_training/inazuma_battle_sd15 \
  --base-model models/anything-v4.5-vae-swapped.safetensors \
  --output-dir outputs/lora_testing/battle_test

# 3. Test Diverse LoRA
python3 scripts/evaluation/test_lora_checkpoints.py \
  outputs/lora_training/inazuma_diverse_sd15 \
  --base-model models/anything-v4.5-vae-swapped.safetensors \
  --output-dir outputs/lora_testing/diverse_test

# 4. Review comparison charts and select best checkpoints
```

## Next Steps

After testing:

1. Review `checkpoint_comparison.png` for visual overview
2. Check `checkpoint_comparison.json` for detailed metrics
3. Select the best-performing checkpoint for each dataset
4. Optionally test top checkpoints with more prompts
5. Deploy selected models for production use
