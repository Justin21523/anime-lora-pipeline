#!/bin/bash
# Quick test script for pipeline components

echo "=========================================="
echo "🎬 Anime Pipeline Quick Test"
echo "=========================================="

# Check Python environment
echo ""
echo "1️⃣  Checking Python environment..."
python3 --version
pip list | grep -E "torch|opencv|transformers" | head -5

# Check CUDA
echo ""
echo "2️⃣  Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Test imports
echo ""
echo "3️⃣  Testing core imports..."
python3 << 'PYEOF'
try:
    import torch
    import cv2
    from PIL import Image
    import numpy as np
    from transformers import CLIPModel, CLIPProcessor
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
PYEOF

# Check directory structure
echo ""
echo "4️⃣  Checking directory structure..."
ls -la scripts/pipeline/*.py 2>/dev/null | wc -l | xargs echo "Pipeline modules found:"

# Test Stage 1 (dry run)
echo ""
echo "5️⃣  Testing Stage 1 module import..."
python3 -c "from scripts.pipeline import stage1_segmentation; print('✅ Stage 1 import successful')" 2>&1 | grep -E "✅|⚠️|❌"

# Test Stage 2a (dry run)
echo ""
echo "6️⃣  Testing Stage 2a module import..."
python3 -c "from scripts.pipeline import stage2a_character_refine; print('✅ Stage 2a import successful')" 2>&1 | grep -E "✅|⚠️|❌"

# Test Orchestrator
echo ""
echo "7️⃣  Testing Orchestrator import..."
python3 -c "from scripts.pipeline import orchestrator; print('✅ Orchestrator import successful')" 2>&1 | grep -E "✅|⚠️|❌"

# Display GPU info if available
echo ""
echo "8️⃣  GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected"

echo ""
echo "=========================================="
echo "✅ Quick test complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install missing dependencies (see docs/PIPELINE_SETUP_AND_USAGE.md)"
echo "  2. Download model weights"
echo "  3. Run: python scripts/pipeline/orchestrator.py --help"
echo ""
