#!/usr/bin/env python3
"""
Interactive Character Sorter for LoRA Training Data
最實用的專業方案：快速手動標註 + 自動相似度推薦

工作流程：
1. 顯示未分類的角色圖片
2. 你為前 5-10 張手動指定角色名稱
3. 系統自動找到相似的圖片並推薦給你確認
4. 重複直到所有重要角色都被標註
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict
from torchvision import models, transforms
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import shutil

class CharacterSorter:
    def __init__(self, device="cuda"):
        """初始化特徵提取器"""
        print("🔧 Loading ResNet50 feature extractor...")
        self.device = device
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✓ Model loaded\n")

    def extract_feature(self, img_path: Path) -> np.ndarray:
        """提取單張圖片的特徵"""
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(img_tensor)
            feature = feature.squeeze().cpu().numpy()

        return feature

    def extract_all_features(self, image_paths: List[Path]) -> Dict:
        """提取所有圖片的特徵"""
        print(f"🔍 Extracting features from {len(image_paths)} images...")

        features = []
        paths = []

        for img_path in tqdm(image_paths):
            try:
                feature = self.extract_feature(img_path)
                features.append(feature)
                paths.append(str(img_path))
            except Exception as e:
                print(f"⚠️  Failed: {img_path.name} - {e}")
                continue

        features_matrix = np.vstack(features)

        # Normalize
        features_matrix = features_matrix / np.linalg.norm(features_matrix, axis=1, keepdims=True)

        return {
            "features": features_matrix,
            "paths": paths
        }

    def find_similar(
        self,
        query_idx: int,
        features_matrix: np.ndarray,
        top_k: int = 20,
        threshold: float = 0.7
    ) -> List[tuple]:
        """找到與查詢圖片相似的圖片"""
        query_feature = features_matrix[query_idx:query_idx+1]
        similarities = cosine_similarity(query_feature, features_matrix)[0]

        # 排除自己
        similarities[query_idx] = -1

        # 找到最相似的
        similar_indices = np.argsort(similarities)[::-1][:top_k]

        # 只返回相似度高於閾值的
        results = []
        for idx in similar_indices:
            sim = similarities[idx]
            if sim >= threshold:
                results.append((idx, sim))

        return results


def create_training_dataset(
    input_dir: Path,
    output_dir: Path,
    annotations_file: Path
):
    """
    根據標註文件創建訓練數據集

    annotations_file 格式:
    {
        "character_name": [
            "path/to/image1.png",
            "path/to/image2.png",
            ...
        ]
    }
    """
    if not annotations_file.exists():
        print(f"❌ Annotations file not found: {annotations_file}")
        return

    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    print(f"\n📁 Creating training dataset in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    for character_name, image_paths in annotations.items():
        char_dir = output_dir / character_name
        char_dir.mkdir(exist_ok=True)

        print(f"  {character_name}: {len(image_paths)} images")

        for img_path_str in image_paths:
            img_path = Path(img_path_str)
            if not img_path.exists():
                continue

            dst_path = char_dir / img_path.name

            # Use hard link for ai_warehouse
            if 'ai_warehouse' in str(output_dir):
                try:
                    import os
                    os.link(img_path, dst_path)
                except OSError:
                    shutil.copy2(img_path, dst_path)
            else:
                shutil.copy2(img_path, dst_path)

            total_images += 1

    print(f"\n✅ Dataset created:")
    print(f"   Characters: {len(annotations)}")
    print(f"   Total images: {total_images}")
    print(f"   Location: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Character Sorter for LoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用流程：

步驟 1: 提取特徵並保存
  python interactive_character_sorter.py extract \\
      /path/to/layered_frames \\
      -o features.npz

步驟 2: 手動標註並查找相似圖片
  （在 Python 中交互式操作）
  >>> from interactive_character_sorter import CharacterSorter
  >>> import numpy as np
  >>>
  >>> data = np.load('features.npz', allow_pickle=True)
  >>> features = data['features']
  >>> paths = data['paths']
  >>>
  >>> sorter = CharacterSorter()
  >>>
  >>> # 找到與第 100 張圖片相似的圖片
  >>> similar = sorter.find_similar(100, features, top_k=20, threshold=0.7)
  >>> for idx, sim in similar:
  ...     print(f"{sim:.3f}: {Path(paths[idx]).name}")

步驟 3: 創建標註文件
  （手動創建 annotations.json）
  {
    "endou_mamoru": ["path1.png", "path2.png", ...],
    "gouenji_shuuya": ["path3.png", "path4.png", ...],
    ...
  }

步驟 4: 生成訓練數據集
  python interactive_character_sorter.py create-dataset \\
      annotations.json \\
      -o /path/to/training_data
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract features from images')
    extract_parser.add_argument('input_dir', type=Path, help='Directory with layered frames')
    extract_parser.add_argument('-o', '--output', type=Path, required=True, help='Output .npz file')
    extract_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # Create dataset command
    create_parser = subparsers.add_parser('create-dataset', help='Create training dataset from annotations')
    create_parser.add_argument('annotations', type=Path, help='Annotations JSON file')
    create_parser.add_argument('-o', '--output-dir', type=Path, required=True, help='Output directory')

    args = parser.parse_args()

    if args.command == 'extract':
        # Find all character images
        all_images = []
        character_dirs = list(args.input_dir.glob("*/character"))

        if character_dirs:
            for char_dir in character_dirs:
                images = list(char_dir.glob("*.png"))
                all_images.extend(images)
        else:
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                all_images.extend(args.input_dir.glob(ext))

        print(f"✓ Found {len(all_images)} images\n")

        if not all_images:
            print("❌ No images found!")
            return

        # Extract features
        sorter = CharacterSorter(device=args.device)
        data = sorter.extract_all_features(all_images)

        # Save
        np.savez_compressed(
            args.output,
            features=data['features'],
            paths=np.array(data['paths'])
        )

        print(f"\n✅ Features saved to {args.output}")
        print("\nNext steps:")
        print("1. Load the features in Python/Jupyter")
        print("2. Manually annotate a few examples per character")
        print("3. Use find_similar() to find more examples")
        print("4. Create annotations.json")
        print("5. Run 'create-dataset' command")

    elif args.command == 'create-dataset':
        create_training_dataset(
            args.input_dir if hasattr(args, 'input_dir') else Path('.'),
            args.output_dir,
            args.annotations
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
