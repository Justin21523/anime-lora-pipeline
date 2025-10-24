# 環境設置指南

## 硬體需求
- GPU: RTX 5080 (CUDA 12.8)
- VRAM: 16GB+
- RAM: 32GB+ 推薦
- 儲存空間: 100GB+ (包含模型和資料集)

## 安裝步驟

### 1. 安裝 PyTorch (RTX 5080 專用)
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 2. 驗證 PyTorch 安裝
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

預期輸出：
```
PyTorch: 2.7.0
CUDA Available: True
CUDA Version: 12.8
```

### 3. 安裝其他依賴
```bash
pip install -r requirements.txt
```

### 4. 安裝 kohya_ss (LoRA 訓練)
```bash
cd models/
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
pip install -r requirements.txt
```

### 5. 下載基礎模型
建議使用的 Stable Diffusion 動漫模型：
- **Anything V5** (推薦)
- **CounterfeitV3**
- **AbyssOrangeMix3**

下載後放置於：`models/base_models/`

### 6. 驗證環境
```bash
python scripts/utils/verify_setup.py
```

## 常見問題

### Q: xformers 安裝失敗
A: 確保 PyTorch 2.7.0 已正確安裝，然後嘗試：
```bash
pip install xformers --no-deps
pip install xformers
```

### Q: CUDA out of memory
A: 調整配置文件中的 batch_size，或啟用 `enable_xformers: true`

### Q: WD14 Tagger 推理緩慢
A: 確認安裝的是 `onnxruntime-gpu` 而非 `onnxruntime`
