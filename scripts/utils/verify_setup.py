"""
Verify installation and setup
"""

import sys
from pathlib import Path
import torch
import torch.version


def check_imports():
    """Check if all required modules can be imported"""
    print("Checking Python packages...")

    required_modules = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "PIL": "Pillow",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "yaml": "PyYAML",
        "omegaconf": "OmegaConf",
        "tqdm": "tqdm",
        "rich": "rich",
        "loguru": "loguru",
        "imagehash": "imagehash",
        "transformers": "transformers",
        "diffusers": "diffusers",
        "onnxruntime": "onnxruntime",
    }

    results = {}
    for module, name in required_modules.items():
        try:
            __import__(module)
            results[name] = True
            print(f"  ✓ {name}")
        except ImportError as e:
            results[name] = False
            print(f"  ✗ {name} - {e}")

    return all(results.values())


def check_pytorch():
    """Check PyTorch and CUDA"""
    print("\nChecking PyTorch...")
    try:
        import torch

        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA version: {torch.cuda}")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_paths():
    """Check warehouse paths"""
    print("\nChecking paths...")

    paths_to_check = {
        "Gold standard": Path("data/characters/endou_mamoru/gold_standard/v1.0/images"),
        "Warehouse training": Path(
            "/mnt/c/AI_LLM_projects/ai_warehouse/training_data/inazuma-eleven"
        ),
        "Warehouse models": Path(
            "/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven"
        ),
        "Warehouse outputs": Path(
            "/mnt/c/AI_LLM_projects/ai_warehouse/outputs/inazuma-eleven"
        ),
        "Base model": Path(
            "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors"
        ),
    }

    results = {}
    for name, path in paths_to_check.items():
        exists = path.exists()
        results[name] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {exists}")

    return all(results.values())


def check_config():
    """Check if config loads correctly"""
    print("\nChecking configuration...")
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from utils.config_loader import load_config, load_character_config

        global_cfg = load_config()
        print(f"  ✓ Global config loaded")

        char_cfg = load_character_config("endou_mamoru")
        print(f"  ✓ Character config loaded")
        print(f"    Character: {char_cfg.character_config.character.name}")
        print(f"    Trigger word: {char_cfg.character_config.character.trigger_word}")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all checks"""
    print("=" * 60)
    print("Inazuma Eleven LoRA Setup Verification")
    print("=" * 60)

    checks = [
        ("Python packages", check_imports),
        ("PyTorch setup", check_pytorch),
        ("File paths", check_paths),
        ("Configuration", check_config),
    ]

    results = {}
    for name, check_func in checks:
        results[name] = check_func()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All checks passed! Setup is complete.")
        return 0
    else:
        print("\n✗ Some checks failed. Please install missing dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
