# Documentation Index (文檔索引)

Complete navigation for Multi-Anime LoRA Training documentation.

---

## 🚀 Quick Access

- [**Quick Start**](QUICKSTART.md) - Get started in 5 minutes
- [**Glossary**](../GLOSSARY.md) - English-Chinese terminology (中英術語對照)
- [**Main README**](../README_NEW.md) - Project overview

---

## 📖 General Guides (通用指南)

Located in `docs/guides/`

### Core Processing
- [Segmentation Guide](guides/segmentation.md) - Character/background separation (分層分割)
- [Clustering Guide](guides/clustering.md) - Character grouping (角色聚類)
- [Training Guide](guides/training.md) - LoRA training basics (LoRA 訓練基礎)
- [Evaluation Guide](guides/evaluation.md) - Quality assessment (品質評估)

### Advanced Features
- [Audio Processing](guides/audio.md) - Audio extraction and analysis (音訊處理)
- [Video Processing](guides/video.md) - Frame extraction and synthesis (影片處理)
- [Effect Detection](guides/effects.md) - Special effects organization (特效檢測)
- [Scene Classification](guides/scenes.md) - Scene type classification (場景分類)
- [ControlNet Preparation](guides/controlnet.md) - Control image generation (ControlNet 準備)
- [Multi-Concept LoRAs](guides/multi_concept.md) - Training multiple concepts (多概念訓練)

---

## 🎬 Anime-Specific Documentation (動畫專屬文檔)

### Yo-kai Watch (妖怪手錶)

Located in `docs/anime-specific/yokai/`

- [**YOKAI_COMPLETE_GUIDE.md**](anime-specific/yokai/YOKAI_COMPLETE_GUIDE.md) - Complete processing guide (完整處理指南)
- [**YOKAI_TOOLS_REFERENCE.md**](anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md) - All tools with examples (工具參考手冊)
- [**YOKAI_SCHEMA.md**](anime-specific/yokai/YOKAI_SCHEMA.md) - Classification taxonomy (分類體系)

**Key Topics**:
- Summon scene detection (召喚場景檢測)
- Soultimate effect extraction (必殺技特效提取)
- Body type classification (60+ types) (身體類型分類)
- Scene hierarchy (realm → location → environment) (場景階層)
- Audio-assisted analysis (音訊輔助分析)

### Inazuma Eleven (閃電十一人)

Located in `docs/anime-specific/inazuma/`

- [**INAZUMA_GUIDE.md**](anime-specific/inazuma/INAZUMA_GUIDE.md) - Getting started (入門指南)
- Coming soon: Keshin detection, soccer action classification

---

## 📚 Reference Documentation (參考文檔)

Located in `docs/reference/`

- [API Reference](reference/api.md) - Code API documentation
- [Model Reference](reference/models.md) - ML models used in project
- [Troubleshooting](reference/troubleshooting.md) - Common issues and solutions
- [Optimization Guide](reference/optimization.md) - Performance tuning

---

## 🎯 By Use Case (依使用情境)

### I want to train a character LoRA (我想訓練角色 LoRA)
1. [Segmentation Guide](guides/segmentation.md) - Extract characters
2. [Clustering Guide](guides/clustering.md) - Group similar characters
3. [Training Guide](guides/training.md) - Train the LoRA

### I want to train an effect LoRA (我想訓練特效 LoRA)
1. [Yokai Summon Detector](anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md#yokai-summon-detector) - Find effect scenes
2. [Effect Organizer](guides/effects.md) - Categorize effects
3. [Training Guide](guides/training.md) - Train effect LoRA

### I want to train a style LoRA (我想訓練風格 LoRA)
1. [Style Classifier](anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md#style-classifier) - Classify characters
2. [Multi-Concept Guide](guides/multi_concept.md) - Prepare grouped training
3. [Training Guide](guides/training.md) - Train style LoRA

### I want to create ControlNet data (我想建立 ControlNet 資料)
1. [Pose Extractor](guides/controlnet.md#pose-extraction) - Extract poses
2. [Depth Generator](guides/controlnet.md#depth-maps) - Generate depth maps
3. [ControlNet Guide](guides/controlnet.md) - Complete pipeline

---

## 🔧 By Tool Type (依工具類型)

### Yokai Watch Tools
- `yokai_summon_detector.py` - [Documentation](anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md#summon-detector)
- `yokai_style_classifier.py` - [Documentation](anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md#style-classifier)
- `scene_type_classifier.py` - [Documentation](anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md#scene-classifier)
- `special_effects_organizer.py` - [Documentation](anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md#effects-organizer)
- `advanced_pose_extractor.py` - [Documentation](anime-specific/yokai/YOKAI_TOOLS_REFERENCE.md#pose-extractor)

### Generic Tools
- `layered_segmentation.py` - [Documentation](guides/segmentation.md)
- `character_clustering.py` - [Documentation](guides/clustering.md)
- `universal_frame_extractor.py` - [Documentation](guides/video.md)
- `audio_extractor.py` - [Documentation](guides/audio.md)

---

## 📊 Status & Coverage

| Category | English Docs | Chinese Terms | Completeness |
|----------|--------------|---------------|--------------|
| Core Guides | ✅ Yes | ✅ In Glossary | 100% |
| Yokai Watch | ✅ Yes | ✅ In Glossary | 100% |
| Inazuma Eleven | 🚧 Partial | ✅ In Glossary | 30% |
| API Reference | 🚧 Coming | ✅ In Glossary | 50% |
| Examples | 🚧 Coming | N/A | 40% |

---

## 🔍 Search by Keyword

### Segmentation (分割)
- [Layered Segmentation Guide](guides/segmentation.md)
- [U2-Net Optimization](reference/optimization.md#u2net)

### Clustering (聚類)
- [Character Clustering Guide](guides/clustering.md)
- [HDBSCAN Parameters](reference/optimization.md#clustering)

### Effects (特效)
- [Effect Detection](guides/effects.md)
- [Yokai Summon Scenes](anime-specific/yokai/YOKAI_COMPLETE_GUIDE.md#summon-effects)
- [Pure Effect Detection](anime-specific/yokai/YOKAI_SCHEMA.md#pure-effects)

### Taxonomy (分類體系)
- [Yokai Schema](anime-specific/yokai/YOKAI_SCHEMA.md)
- [Scene Types](anime-specific/yokai/YOKAI_SCHEMA.md#scene-taxonomy)
- [Body Types](anime-specific/yokai/YOKAI_SCHEMA.md#body-types)

### Training (訓練)
- [Basic Training Guide](guides/training.md)
- [Multi-Concept LoRAs](guides/multi_concept.md)
- [Training Parameters](reference/optimization.md#training)

---

## 💡 Tips

- **New to the project?** Start with [Quick Start](QUICKSTART.md)
- **Looking for terminology?** Check [GLOSSARY.md](../GLOSSARY.md)
- **Anime-specific features?** Go to [anime-specific/](anime-specific/) folder
- **Troubleshooting?** See [reference/troubleshooting.md](reference/troubleshooting.md)

---

**Last Updated**: 2025-10-30
**Documentation Version**: 2.0
**Status**: Phase 1 Complete (Framework established)

---

## 📝 Document Status Legend

- ✅ **Complete** - Fully written and reviewed
- 🚧 **In Progress** - Being written or updated
- 📝 **Planned** - Scheduled for future addition
- ⚠️ **Needs Update** - Outdated, requires revision
