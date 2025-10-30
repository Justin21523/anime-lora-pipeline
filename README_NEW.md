# Multi-Anime LoRA Training Toolkit

> **多動畫 LoRA 訓練工具包**

Professional toolkit for training character and style LoRAs from anime series.

Currently supports: **Yo-kai Watch (妖怪手錶)** | **Inazuma Eleven (閃電十一人)**

---

## 📚 Quick Links (快速連結)

- [**Quick Start Guide**](docs/QUICKSTART.md) - Get started in 5 minutes
- [**Glossary**](GLOSSARY.md) - English-Chinese terminology reference (中英術語對照)
- [**Documentation Index**](docs/INDEX.md) - Complete documentation navigation
- [**Yokai Watch Guide**](docs/anime-specific/yokai/) - Yo-kai Watch specific documentation
- [**Inazuma Eleven Guide**](docs/anime-specific/inazuma/) - Inazuma Eleven specific documentation

---

## ✨ Features

- **Multi-Series Support**: Organized structure for multiple anime series
- **Advanced Segmentation**: U2-Net, YOLO, Mask2Former for character extraction
- **Smart Clustering**: HDBSCAN + CLIP for automatic character grouping
- **Effect Detection**: Specialized tools for summon scenes, attacks, transformations
- **Taxonomy-Based**: Hierarchical classification (scenes, body types, effects)
- **Complete Pipelines**: Automated workflows from video → training data
- **ControlNet Support**: Pose, depth, and multi-modal control image generation
- **Bilingual Docs**: English documentation with Chinese terminology reference

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements/all.txt

# Or install by category
pip install -r requirements/core.txt
pip install -r requirements/segmentation.txt  # Optional: if using segmentation
pip install -r requirements/clustering.txt     # Optional: if using clustering
```

### 2. Choose Your Anime

#### Yo-kai Watch (妖怪手錶)
```bash
# Run full processing pipeline
python scripts/yokai/pipelines/yokai_pipeline_standard.py \
    --input /path/to/yokai/episodes \
    --output /path/to/training_data

# See complete guide
cat docs/anime-specific/yokai/YOKAI_COMPLETE_GUIDE.md
```

#### Inazuma Eleven (閃電十一人)
```bash
# Processing pipeline (coming soon)
python scripts/inazuma/pipelines/inazuma_pipeline.py \
    --input /path/to/inazuma/episodes \
    --output /path/to/training_data

# See guide
cat docs/anime-specific/inazuma/INAZUMA_GUIDE.md
```

---

## 📂 Project Structure

```
multi-anime-lora-training/
├── scripts/
│   ├── core/              # Shared utilities and models
│   ├── yokai/             # Yo-kai Watch specific tools
│   ├── inazuma/           # Inazuma Eleven specific tools
│   ├── generic/           # Anime-agnostic processing tools
│   ├── evaluation/        # Quality assessment
│   └── tests/             # Test suites
│
├── docs/
│   ├── guides/            # General processing guides
│   ├── anime-specific/    # Anime-specific documentation
│   └── reference/         # Technical reference
│
├── configs/               # Training configurations
├── requirements/          # Dependency specifications
└── GLOSSARY.md           # Terminology reference (中英對照)
```

---

## 🎯 What Can You Train?

### Character LoRAs
- Individual characters with consistent style
- Multiple characters from same series
- Character variants (transformations, outfits)

### Style LoRAs
- Body type styles (humanoid, beast, mecha, etc.)
- Attribute styles (fire, water, wind, etc.)
- Art styles (cute, cool, brave, etc.)

### Effect LoRAs
- Summon effects (magic circles, beams, portals)
- Attack effects (beams, auras, explosions)
- Transformation effects (shadowside, fusion, etc.)

### Background LoRAs
- Scene types (school, forest, yo-kai world, etc.)
- Time variations (day, night, festival, etc.)
- Environment types (indoor, outdoor, fantasy, etc.)

### ControlNet Datasets
- Character poses (humanoid and non-humanoid)
- Depth maps (optimized for anime)
- Line art and segmentation masks

---

## 📖 Documentation

### Getting Started
- [Quick Start](docs/QUICKSTART.md) - 5-minute setup
- [Installation Guide](docs/guides/installation.md)
- [First Training](docs/guides/first_training.md)

### Core Concepts
- [Segmentation Guide](docs/guides/segmentation.md) - Character extraction
- [Clustering Guide](docs/guides/clustering.md) - Character grouping
- [Training Guide](docs/guides/training.md) - LoRA training basics

### Anime-Specific
- [Yo-kai Watch Complete Guide](docs/anime-specific/yokai/YOKAI_COMPLETE_GUIDE.md)
- [Yo-kai Watch Tools Reference](docs/anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md)
- [Yo-kai Watch Schema](docs/anime-specific/yokai/YOKAI_SCHEMA.md)
- [Inazuma Eleven Guide](docs/anime-specific/inazuma/INAZUMA_GUIDE.md)

### Advanced Topics
- [Effect Detection & Organization](docs/guides/effects.md)
- [Scene Classification](docs/guides/scenes.md)
- [ControlNet Preparation](docs/guides/controlnet.md)
- [Multi-Concept LoRAs](docs/guides/multi_concept.md)

---

## 🛠️ Tools Overview

### Yokai Watch Tools (妖怪手錶工具)
- `yokai_summon_detector.py` - Detect summon scenes (召喚場景檢測)
- `yokai_style_classifier.py` - AI style classification (AI 風格分類)
- `scene_type_classifier.py` - Hierarchical scene classification (場景分類)
- `special_effects_organizer.py` - Effect categorization (特效組織)
- `advanced_pose_extractor.py` - Pose detection (姿態提取)

### Generic Tools (通用工具)
- `layered_segmentation.py` - Character/background separation (分層分割)
- `character_clustering.py` - Automatic character grouping (角色聚類)
- `universal_frame_extractor.py` - Video frame extraction (影格提取)
- `audio_extractor.py` - Audio analysis and extraction (音訊分析)

### Evaluation Tools (評估工具)
- `lora_validator.py` - Quality metrics (品質評估)
- `compare_models.py` - Model comparison (模型比較)

---

## 💻 Requirements

- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA support
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **Storage**: 50GB+ free space for processing
- **RAM**: 16GB+ (32GB recommended)

---

## 📊 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Infrastructure | ✅ Complete | Segmentation, clustering, pipelines |
| Yo-kai Watch Support | ✅ Complete | 16 specialized tools, full taxonomy |
| Inazuma Eleven Support | 🚧 In Progress | Basic structure, needs tools |
| Documentation | ✅ Complete | English docs + terminology |
| Test Suite | ✅ Complete | 4 consolidated test files |

---

## 🤝 Contributing

This is a research project. For issues or suggestions, please check the documentation first.

---

## 📜 License

[Add your license here]

---

## 🙏 Credits

- **U2-Net**: Qin et al., Xueb in Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane, Martin Jagersand
- **CLIP**: OpenAI
- **BLIP-2**: Salesforce Research
- **YOLO**: Ultralytics
- **HDBSCAN**: Leland McInnes

---

## 📧 Contact

[Add contact information]

---

**Version**: 2.0.0
**Last Updated**: 2025-10-30
**Project Reorganization**: Phase 1 Complete

---

## 🌏 Language / 語言

This project uses English as the primary language for all documentation and code, with Chinese terminology provided in the [GLOSSARY.md](GLOSSARY.md) file.

本專案使用英文作為所有文檔和程式碼的主要語言，中文術語對照請參考 [GLOSSARY.md](GLOSSARY.md) 檔案。
