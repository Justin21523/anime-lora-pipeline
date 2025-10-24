inazuma-eleven-lora/
│
├── README.md                          # 專案總覽與快速開始
├── ARCHITECTURE.md                    # 架構設計文檔
├── requirements.txt                   # Python 依賴
├── .gitignore                         # Git 忽略規則
├── .env.example                       # 環境變數範本
│
├── config/                            # 🔧 配置檔案中心
│   ├── global_config.yaml            # 全域設定（路徑、模型參數）
│   ├── characters/                    # 各角色專屬配置
│   │   ├── endou_mamoru.yaml
│   │   ├── gouenji_shuuya.yaml
│   │   ├── fudou_akio.yaml
│   │   └── utsunomiya_toramaru.yaml
│   └── training_presets.yaml          # LoRA 訓練預設組合
│
├── data/                              # 📦 資料倉儲（AI Warehouse）
│   ├── raw_videos/                    # 原始影片
│   │   ├── S01E01.mkv
│   │   └── S01E02.mkv
│   │
│   ├── processed_frames/              # 影格快取
│   │   └── S01E01/
│   │       ├── frame_0001.jpg
│   │       └── metadata.json
│   │
│   ├── characters/                    # 各角色資料集
│   │   ├── endou_mamoru/
│   │   │   ├── gold_standard/        # 手動挑選的黃金標準
│   │   │   │   ├── v1.0/             # 版本管理
│   │   │   │   │   ├── images/
│   │   │   │   │   └── metadata.csv
│   │   │   │   └── v1.5/
│   │   │   │
│   │   │   ├── auto_collected/       # 自動收集的資料
│   │   │   │   ├── raw/              # 原始篩選結果
│   │   │   │   ├── cleaned/          # 清理後
│   │   │   │   └── rejected/         # 被過濾掉的（供檢討）
│   │   │   │
│   │   │   └── training_ready/       # 最終訓練資料集
│   │   │       ├── images/
│   │   │       ├── captions/         # .txt 標註檔
│   │   │       └── dataset_info.json
│   │   │
│   │   ├── gouenji_shuuya/
│   │   ├── fudou_akio/
│   │   └── utsunomiya_toramaru/
│   │
│   └── cache/                         # 臨時快取
│       ├── clip_embeddings/           # CLIP 嵌入快取
│       ├── hash_index/                # 圖片哈希索引
│       └── face_detections/           # 人臉檢測快取
│
├── models/                            # 🧠 模型倉庫
│   ├── base_models/                   # Base SD 模型
│   │   ├── sd15_anime.safetensors
│   │   └── sdxl_anime.safetensors
│   │
│   ├── loras/                         # 訓練好的 LoRA
│   │   ├── endou_mamoru/
│   │   │   ├── v0.1_poc/
│   │   │   │   ├── endou_v0.1.safetensors
│   │   │   │   ├── training_log.txt
│   │   │   │   └── hyperparameters.json
│   │   │   ├── v1.0_auto/
│   │   │   └── v1.5_optimized/
│   │   │
│   │   ├── gouenji_shuuya/
│   │   ├── fudou_akio/
│   │   └── utsunomiya_toramaru/
│   │
│   └── checkpoints/                   # 訓練中間檢查點
│
├── scripts/                           # 🛠️ 核心腳本（模組化）
│   ├── __init__.py
│   │
│   ├── pipeline/                      # 自動化管線
│   │   ├── __init__.py
│   │   ├── video_processor.py        # Module 1: 影片處理
│   │   ├── image_cleaner.py          # Module 2: 圖片清理
│   │   ├── character_filter.py       # Module 3: 角色過濾
│   │   ├── auto_captioner.py         # Module 4: 自動標註
│   │   └── pipeline_orchestrator.py  # 管線編排器
│   │
│   ├── training/                      # LoRA 訓練
│   │   ├── __init__.py
│   │   ├── train_lora.py             # 訓練腳本
│   │   ├── resume_training.py        # 恢復訓練
│   │   └── multi_char_trainer.py     # 多角色批次訓練
│   │
│   ├── evaluation/                    # 評估系統
│   │   ├── __init__.py
│   │   ├── quality_evaluator.py      # 品質評估（CLIP、LPIPS）
│   │   ├── generate_test_images.py   # 生成測試圖
│   │   └── comparison_report.py      # 版本對比報告
│   │
│   ├── utils/                         # 工具函數
│   │   ├── __init__.py
│   │   ├── clip_utils.py             # CLIP 相關
│   │   ├── hash_utils.py             # 圖片哈希
│   │   ├── file_manager.py           # 檔案管理
│   │   ├── config_loader.py          # 配置載入
│   │   └── logger.py                 # 日誌系統
│   │
│   └── tools/                         # 獨立工具腳本
│       ├── validate_gold_standard.py  # 驗證黃金標準圖片
│       ├── analyze_dataset.py        # 資料集分析報告
│       ├── migrate_character.py      # 遷移角色資料
│       └── cleanup_cache.py          # 清理快取
│
├── notebooks/                         # 📊 Jupyter 筆記本（實驗用）
│   ├── 01_data_exploration.ipynb
│   ├── 02_clip_similarity_test.ipynb
│   └── 03_lora_comparison.ipynb
│
├── outputs/                           # 📤 輸出結果
│   ├── evaluation_reports/           # 評估報告
│   │   └── endou_v1.0_vs_v0.1.html
│   │
│   ├── generated_images/             # 生成的測試圖
│   │   └── endou_mamoru/
│   │       └── 2024-01-15_test_matrix/
│   │
│   └── logs/                         # 運行日誌
│       ├── pipeline_2024-01-15.log
│       └── training_2024-01-15.log
│
├── tests/                            # 🧪 單元測試
│   ├── __init__.py
│   ├── test_video_processor.py
│   ├── test_character_filter.py
│   └── test_quality_evaluator.py
│
└── docs/                             # 📚 詳細文檔
    ├── setup_guide.md                # 環境設置指南
    ├── pipeline_workflow.md          # 管線工作流程
    ├── character_guide.md            # 角色配置指南
    ├── training_guide.md             # 訓練指南
    └── troubleshooting.md            # 常見問題排解