#!/usr/bin/env python3
"""
圖片分析與重新命名工具
分析円堂守的照片，根據姿勢、情緒、場景等特徵重新命名
並篩選出符合黃金標準的照片
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

# 圖片目錄
IMAGE_DIR = Path("/mnt/c/AI_LLM_projects/inazuma-eleven-lora/data/characters/endou_mamoru/images")
OUTPUT_JSON = Path("/mnt/c/AI_LLM_projects/inazuma-eleven-lora/data/characters/endou_mamoru/image_analysis.json")

def get_all_images() -> List[Path]:
    """獲取所有圖片檔案"""
    extensions = ['.png', '.jpg', '.jpeg', '.webp']
    images = []
    for ext in extensions:
        images.extend(IMAGE_DIR.glob(f'*{ext}'))
    return sorted(images)

def create_image_catalog() -> Dict:
    """創建圖片目錄"""
    images = get_all_images()

    catalog = {
        "total_count": len(images),
        "images": []
    }

    for idx, img_path in enumerate(images, 1):
        catalog["images"].append({
            "index": idx,
            "original_name": img_path.name,
            "path": str(img_path),
            "analyzed": False,
            "new_name": "",
            "features": {
                "pose": "",
                "expression": "",
                "outfit": "",
                "scene": "",
                "angle": "",
                "special_notes": ""
            },
            "quality_assessment": {
                "clarity": "",  # clear/slightly_blurry/blurry
                "composition": "",  # good/average/poor
                "character_visibility": "",  # full/partial/distant
                "is_solo": True,
                "has_signature_items": []  # headband, gloves, etc
            },
            "gold_standard_candidate": False,
            "rejection_reason": ""
        })

    return catalog

def save_catalog(catalog: Dict):
    """保存目錄到JSON"""
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"目錄已保存到: {OUTPUT_JSON}")

def main():
    """主函數"""
    print("=== 円堂守圖片分析工具 ===")
    print(f"掃描目錄: {IMAGE_DIR}")

    # 創建圖片目錄
    catalog = create_image_catalog()

    print(f"\n找到 {catalog['total_count']} 張圖片")
    print("\n前10張圖片:")
    for img in catalog['images'][:10]:
        print(f"  {img['index']:3d}. {img['original_name']}")

    # 保存目錄
    save_catalog(catalog)

    print("\n下一步:")
    print("1. 使用 Claude 逐一查看圖片並填寫分析資訊")
    print("2. 根據分析結果生成新檔名")
    print("3. 篩選黃金標準候選照片")
    print("4. 執行重新命名和複製操作")

if __name__ == "__main__":
    main()
