# Anime LoRA Pipeline

A modular, end-to-end pipeline for training **anime character LoRA models** with Stable Diffusion.
From raw anime videos to high-quality LoRA training data — fully automated with flexible configs.

---

## ✨ Features

- 🎬 **Video Frame Extraction**: Detect and extract keyframes using `PySceneDetect`.
- 🧹 **Image Cleaning**: Deduplication, blur/brightness checks, resolution filters.
- 🧑‍🤝‍🧑 **Character Filtering**: Two-stage filtering with **WD14 Tagger** + **CLIP similarity**.
- 📝 **Auto-Captioning**: Automated caption generation with blacklist/tag control.
- 🔄 **Pipeline Orchestrator**: End-to-end automation with error handling & reports.
- 🧪 **Evaluation Tools**: Generate test images, compute CLIP/LPIPS metrics, compare versions.
- ⚙️ **Warehouse System**: Shared storage for datasets, base models, LoRA outputs, and cache.

---

## 📂 Project Structure

```bash
anime-lora-pipeline/
├── config/                # Global + character-specific configs
├── data/                  # Datasets (symlinked to AI Warehouse)
├── models/                # Base models + trained LoRAs
├── scripts/               # Core pipeline + training modules
├── outputs/               # Logs, reports, generated samples
├── notebooks/             # Jupyter notebooks for experiments
└── docs/                  # Detailed guides & troubleshooting
````

For the full structure, see [project_structure.md](docs/project_structure.md).

---

## 🚀 Quick Start

### 1. Setup Environment

Hardware requirements:

* **GPU**: RTX 5080 (16GB VRAM)
* **RAM**: 32GB+
* **Disk**: 100GB+

Install dependencies:

```bash
# Install PyTorch (CUDA 12.8 for RTX 5080)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt
```

Clone and set up **kohya_ss** for LoRA training:

```bash
cd models/
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
pip install -r requirements.txt
```

For detailed setup: [SETUP.md](docs/setup_guide.md)

---

### 2. Prepare Training Data

#### Option A: Use Existing Character (Recommended)

```bash
python scripts/tools/prepare_training_data.py endou_mamoru --style detailed --repeat 10
```

This creates a `training_ready/` dataset with images + captions.

#### Option B: Run Full Pipeline (New Characters)

```bash
python scripts/pipeline/pipeline_orchestrator.py gouenji_shuuya --videos data/raw_videos/S01E01.mkv
```

---

### 3. Train LoRA

```bash
cd models/sd-scripts
python train_network.py --network_module=networks.lora --pretrained_model_name_or_path=../base_models/anything-v4.5-vae-swapped.safetensors --train_data_dir=../../data/characters/endou_mamoru/training_ready/ --output_dir=../../models/loras/endou_mamoru/v1.0_auto --resolution=512,512 --batch_size=4 --learning_rate=1e-4 --max_train_steps=20000
```

*(Hyperparameters are configurable in `config/training_presets.yaml`.)*

---

## 📊 Evaluation

Generate test images and compare LoRA versions:

```bash
python scripts/evaluation/generate_test_images.py endou_mamoru
python scripts/evaluation/comparison_report.py endou_mamoru --versions v0.1 v1.0
```

Reports are saved in `outputs/evaluation_reports/`.

---

## 🏗️ AI Warehouse Integration

This project uses a **shared warehouse system** for storage efficiency and cross-project reuse.
All datasets, models, and outputs are symlinked to a centralized path.

See: [WAREHOUSE_SETUP.md](docs/warehouse_setup.md)

---

## 📖 Documentation

* [Setup Guide](docs/setup_guide.md)
* [Pipeline Workflow](docs/pipeline_workflow.md)
* [Character Guide](docs/character_guide.md)
* [Training Guide](docs/training_guide.md)
* [Troubleshooting](docs/troubleshooting.md)

---

## 🔮 Roadmap

* ✅ Automated data pipeline (complete)
* 🚧 LoRA baseline training
* 🔜 Multi-character batch training
* 🔜 Automated evaluation system
* 🔜 Community release on Civitai

---

## 📜 License

MIT License. Free to use, modify, and share.
(Please respect copyright of original anime assets.)

---

## 🎓 References

* [WD14 Tagger](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3)
* [CLIP](https://github.com/openai/CLIP)
* [kohya_ss](https://github.com/kohya-ss/sd-scripts)
* [LoRA Training Guide](https://rentry.org/lora_train)

---

**Next milestone:** Generate captions for gold-standard images and run the first LoRA baseline training 🚀

