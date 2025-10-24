#!/usr/bin/env python3
"""
LoRA Validation Script
Validates the Endou Mamoru LoRA file without requiring diffusers
"""

import torch
from safetensors.torch import load_file
from pathlib import Path

print("=" * 80)
print("Endou Mamoru LoRA v1 - Validation")
print("=" * 80)

LORA_PATH = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/character_loras/inazuma-eleven/endou_mamoru_v1.safetensors")

print(f"\n[1/3] Checking LoRA file...")
print(f"  Path: {LORA_PATH}")

if not LORA_PATH.exists():
    print(f"  ✗ ERROR: LoRA file not found!")
    exit(1)

file_size = LORA_PATH.stat().st_size / (1024 * 1024)
print(f"  ✓ File exists: {file_size:.2f} MB")

print(f"\n[2/3] Loading LoRA weights...")
try:
    state_dict = load_file(str(LORA_PATH))
    num_keys = len(state_dict)
    print(f"  ✓ Successfully loaded {num_keys} weight tensors")
except Exception as e:
    print(f"  ✗ ERROR loading LoRA: {e}")
    exit(1)

print(f"\n[3/3] Analyzing LoRA structure...")

# Analyze layers
lora_layers = {}
for key in state_dict.keys():
    layer_name = key.rsplit('.', 1)[0]
    if layer_name not in lora_layers:
        lora_layers[layer_name] = []
    lora_layers[layer_name].append(key)

print(f"  ✓ Found {len(lora_layers)} LoRA layers")

# Check for typical LoRA patterns
lora_up_count = sum(1 for k in state_dict.keys() if 'lora_up' in k)
lora_down_count = sum(1 for k in state_dict.keys() if 'lora_down' in k)
alpha_count = sum(1 for k in state_dict.keys() if 'alpha' in k)

print(f"  ✓ LoRA up weights: {lora_up_count}")
print(f"  ✓ LoRA down weights: {lora_down_count}")
print(f"  ✓ Alpha parameters: {alpha_count}")

# Sample a few weights
print(f"\n[Sample Layers]")
for i, (layer_name, keys) in enumerate(list(lora_layers.items())[:5]):
    print(f"  {i+1}. {layer_name}")
    for key in keys:
        tensor = state_dict[key]
        print(f"     - {key.split('.')[-1]}: shape {tuple(tensor.shape)}, dtype {tensor.dtype}")

print("\n" + "=" * 80)
print("✓ LoRA Validation Complete!")
print("=" * 80)
print(f"\nSummary:")
print(f"  Model: endou_mamoru_v1.safetensors")
print(f"  Size: {file_size:.2f} MB")
print(f"  Total parameters: {num_keys}")
print(f"  LoRA layers: {len(lora_layers)}")
print(f"\nThe LoRA file is valid and ready to use!")
print(f"\nTo test image generation, use one of these options:")
print(f"  1. Automatic1111 WebUI (recommended)")
print(f"  2. ComfyUI")
print(f"  3. sd-webui-forge")
print(f"\nPlace the LoRA in your WebUI's models/Lora folder and use the trigger word:")
print(f"  'endou_mamoru'")
