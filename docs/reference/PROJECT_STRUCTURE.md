# Project Structure

## Overview
This document describes the recommended directory structure for the Inazuma Eleven LoRA training project.

## Directory Layout

```
/mnt/c/AI_LLM_projects/
│
├── inazuma-eleven-lora/              # Main project repository
│   ├── scripts/
│   │   ├── batch/                    # Batch processing scripts
│   │   │   ├── run_character_clustering.sh
│   │   │   ├── segment_all_episodes.sh
│   │   │   ├── auto_train_sequence.sh
│   │   │   └── ...
│   │   ├── tools/                    # Individual tool scripts
│   │   │   ├── character_clustering.py
│   │   │   ├── frame_interpolator.py
│   │   │   ├── layered_segmentation.py
│   │   │   └── ...
│   │   └── evaluation/               # Quality evaluation scripts
│   │       ├── test_lora_checkpoints.py
│   │       ├── lora_quality_metrics.py
│   │       └── compare_lora_models.py
│   ├── configs/                      # Training configurations
│   │   ├── train_diverse_sd15_updated.toml
│   │   ├── train_battle_sd15.toml
│   │   ├── dataset_diverse_sd15.toml
│   │   └── dataset_battle_sd15.toml
│   ├── docs/                         # Documentation
│   │   ├── PROJECT_STRUCTURE.md      # This file
│   │   ├── CHARACTER_CLUSTERING_GUIDE.md
│   │   ├── LORA_TESTING_GUIDE.md
│   │   └── ...
│   ├── prompts/                      # Prompt templates
│   ├── examples/                     # Example scripts and configs
│   ├── outputs/                      # Project outputs (NEW!)
│   │   ├── lora_testing/            # LoRA checkpoint test results
│   │   │   ├── battle_sd15_test/
│   │   │   └── diverse_sd15_test/
│   │   └── experiments/             # Experimental results
│   └── README.md
│
└── ai_warehouse/                     # Data warehouse (all actual data)
    ├── cache/                        # Intermediate processing results
    │   └── inazuma-eleven/
    │       ├── extracted_frames/     # Raw extracted frames
    │       ├── interpolated_frames/  # Frame interpolation output
    │       └── layered_frames/       # Segmented layers (character/background/masks)
    │           ├── episode_001/
    │           │   ├── character/    # Character layer (PNG with alpha)
    │           │   ├── background/   # Background layer (JPG)
    │           │   └── masks/        # Binary masks
    │           └── episode_002/
    │               └── ...
    │
    ├── training_data/                # Prepared training datasets
    │   └── inazuma-eleven/
    │       └── character_clusters/   # Clustered characters (hard links)
    │           ├── character_000/    # Main character
    │           ├── character_001/    # Supporting character
    │           ├── ...
    │           └── noise/            # Outliers
    │
    ├── models/                       # Trained models and LoRAs
    │   ├── lora/
    │   │   └── character_loras/
    │   │       └── inazuma-eleven/
    │   │           ├── endou_mamoru_v1.safetensors
    │   │           └── ...
    │   └── stable-diffusion/
    │       └── anything-v4.5-vae-swapped.safetensors
    │
    └── outputs/                      # Large-scale processing outputs
        ├── character_clustering/     # Clustering outputs (deprecated)
        │   ├── logs/                 # Clustering log files
        │   ├── tests/                # Test runs
        │   └── comparisons/          # Method comparisons
        └── lora_training/            # Trained LoRA models
            ├── inazuma_diverse_sd15/ # Diverse dataset LoRA
            └── inazuma_battle_sd15/  # Battle dataset LoRA
```

## Design Principles

### 1. Project Outputs vs. Data Warehouse

**Project Outputs** (`inazuma-eleven-lora/outputs/`):
- LoRA testing and evaluation results
- Experimental results and comparisons
- Quality metrics and visualizations
- Generated test images and reports
- Relatively small files, frequently accessed
- Version-controlled friendly (JSON, markdown, small PNGs)

**Data Warehouse** (`ai_warehouse/`):
- Large-scale source data (videos, images)
- Intermediate processing results (frames, layers)
- Training datasets (thousands of images)
- Trained model files (LoRA safetensors)
- Large binary files, infrequently modified
- NOT version-controlled (too large, use separate backup)

**Rule**: Keep testing/analysis outputs in project folder, keep training data/models in warehouse.

### 2. No Temporary Files in /tmp
**Rule**: All project files, scripts, and data should reside in the project directories, NOT in `/tmp`.

**Reasons**:
- `/tmp` is cleared on system restart
- Better organization and version control
- Easier to locate and manage files
- Consistent backup and archival

**Exception**: Only truly temporary files (like single-use test outputs) may use `/tmp`.

### 2. Separation of Code and Data

**Code**: `inazuma-eleven-lora/`
- Python scripts
- Bash scripts
- Documentation
- Configuration files
- Version controlled with git

**Data**: `ai_warehouse/`
- All actual data files (videos, images, models)
- Intermediate processing results
- Final training datasets
- NOT version controlled (too large)
- Backed up separately

### 3. Hard Links for ai_warehouse

**Policy**: All files in `ai_warehouse` directories must be actual files or hard links (NOT symbolic links).

**Implementation**: Tools automatically detect `ai_warehouse` in output paths and use hard links.

**Benefits**:
- Windows compatibility
- No data duplication
- Single source of truth
- Reliable backups

## Script Locations

### Batch Processing Scripts
Location: `inazuma-eleven-lora/scripts/batch/`

| Script | Purpose |
|--------|---------|
| `run_character_clustering.sh` | Run character clustering with hard links |
| `segment_all_episodes.sh` | Batch segmentation of all episodes |

### Tool Scripts
Location: `inazuma-eleven-lora/scripts/tools/`

| Script | Purpose |
|--------|---------|
| `character_clustering.py` | CLIP + HDBSCAN clustering |
| `frame_interpolator.py` | Frame interpolation for quality |
| `layered_segmentation.py` | Character/background separation |
| `universal_frame_extractor.py` | Extract frames from videos |

## Data Flow

```
1. Video Files (source)
   ↓
2. cache/extracted_frames/        (frame extraction)
   ↓
3. cache/interpolated_frames/     (optional: frame interpolation)
   ↓
4. cache/layered_frames/          (character segmentation)
   ↓
5. training_data/character_clusters/  (character clustering with hard links)
   ↓
6. models/lora/                   (LoRA training output)
```

## Log Files

All logs should be saved to permanent locations:

- **Clustering logs**: `ai_warehouse/outputs/character_clustering/logs/`
- **Segmentation logs**: `ai_warehouse/outputs/segmentation/logs/`
- **Training logs**: `ai_warehouse/outputs/training/logs/`

**Format**: `{operation}_YYYYMMDD_HHMMSS.log`

## Quick Start Commands

### Run Character Clustering
```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora
bash scripts/batch/run_character_clustering.sh
```

### Direct Python Execution
```bash
cd /mnt/c/AI_LLM_projects/inazuma-eleven-lora

conda run -n blip2-env python3 scripts/tools/character_clustering.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/cache/inazuma-eleven/layered_frames \
  --output-dir /mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven/character_clusters \
  --min-cluster-size 25 \
  --device cuda
```

## Best Practices

1. **Always use absolute paths** in scripts for clarity
2. **Use variables** for common paths (PROJECT_ROOT, WAREHOUSE_ROOT)
3. **Save logs** to `ai_warehouse/outputs/` with timestamps
4. **Document new scripts** in this file when adding them
5. **Test with small datasets** before running on full data
6. **Verify file accessibility** in Windows after processing

## Testing and Evaluation Workflow

LoRA testing outputs are stored in the **project folder** for easy access:

```bash
# Test LoRA checkpoints (outputs go to project folder)
python3 scripts/evaluation/test_lora_checkpoints.py \
  /mnt/c/AI_LLM_projects/ai_warehouse/outputs/lora_training/inazuma_battle_sd15 \
  --base-model /path/to/base/model.safetensors \
  --output-dir outputs/lora_testing/battle_test \
  --device cuda
```

Results location: `inazuma-eleven-lora/outputs/lora_testing/`

See `docs/LORA_TESTING_GUIDE.md` for complete testing documentation.

## Version History

- **2025-10-27**: Initial structure documentation
- **2025-10-27**: Added hard link policy for ai_warehouse
- **2025-10-27**: Moved scripts from /tmp to project directories
- **2025-10-30**: Added `outputs/` directory for project results
- **2025-10-30**: Clarified separation between project outputs and data warehouse
