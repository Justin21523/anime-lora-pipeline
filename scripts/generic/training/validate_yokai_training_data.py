#!/usr/bin/env python3
"""
Training Data Validator for Yokai Watch LoRA

Validates training data before starting LoRA training:
- Image integrity and quality checks
- Caption file validation
- Directory structure verification
- Filename pairing validation
- kohya_ss format compliance
- Quality metrics and recommendations

Integrates checks from existing tools:
- image_cleaner.py quality checks
- character_filter.py filtering logic
"""

from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class TrainingDataValidator:
    """Validates LoRA training data"""

    def __init__(self, min_resolution: int = 512, max_aspect_ratio: float = 3.0):
        """
        Initialize validator

        Args:
            min_resolution: Minimum image dimension (default: 512)
            max_aspect_ratio: Maximum aspect ratio (default: 3.0)
        """
        self.min_resolution = min_resolution
        self.max_aspect_ratio = max_aspect_ratio

    def check_image_quality(self, img_path: Path) -> Dict:
        """
        Check image quality and integrity

        Args:
            img_path: Path to image

        Returns:
            Quality check results
        """
        issues = []
        warnings = []

        try:
            # Try to open image
            img = Image.open(img_path)

            # Check mode
            if img.mode not in ['RGB', 'RGBA']:
                issues.append(f"Invalid mode: {img.mode} (expected RGB or RGBA)")

            # Check dimensions
            width, height = img.size

            if width < self.min_resolution or height < self.min_resolution:
                issues.append(
                    f"Resolution too low: {width}x{height} "
                    f"(minimum: {self.min_resolution}px)"
                )

            # Check aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > self.max_aspect_ratio:
                warnings.append(
                    f"Extreme aspect ratio: {aspect_ratio:.2f} "
                    f"(may cause issues during training)"
                )

            # Check if image is too small in file size
            file_size = img_path.stat().st_size
            if file_size < 10_000:  # < 10KB
                warnings.append(f"Very small file size: {file_size} bytes")

            # Check if image is completely blank
            extrema = img.convert('L').getextrema()
            if extrema[0] == extrema[1]:  # All pixels same value
                issues.append("Image is completely blank")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "width": width,
                "height": height,
                "mode": img.mode,
                "file_size": file_size
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Failed to open image: {str(e)}"],
                "warnings": [],
                "width": None,
                "height": None,
                "mode": None,
                "file_size": None
            }

    def check_caption(self, caption_path: Path) -> Dict:
        """
        Check caption file

        Args:
            caption_path: Path to caption file

        Returns:
            Caption check results
        """
        issues = []
        warnings = []

        if not caption_path.exists():
            return {
                "valid": False,
                "issues": ["Caption file missing"],
                "warnings": [],
                "caption": None,
                "length": None
            }

        try:
            # Read caption
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()

            # Check if empty
            if not caption:
                issues.append("Caption is empty")

            # Check length
            if len(caption) < 10:
                warnings.append(f"Very short caption: {len(caption)} chars")
            elif len(caption) > 300:
                warnings.append(f"Very long caption: {len(caption)} chars")

            # Check if caption is just filename
            if caption.lower().replace('.png', '').replace('.jpg', '') in str(caption_path.stem).lower():
                warnings.append("Caption might be just filename")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "caption": caption,
                "length": len(caption)
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Failed to read caption: {str(e)}"],
                "warnings": [],
                "caption": None,
                "length": None
            }

    def validate_training_pair(self, img_path: Path) -> Dict:
        """
        Validate image + caption pair

        Args:
            img_path: Path to image file

        Returns:
            Validation results for the pair
        """
        caption_path = img_path.with_suffix('.txt')

        # Check image
        img_result = self.check_image_quality(img_path)

        # Check caption
        caption_result = self.check_caption(caption_path)

        # Combined validation
        all_issues = img_result["issues"] + caption_result["issues"]
        all_warnings = img_result["warnings"] + caption_result["warnings"]

        return {
            "image_path": str(img_path),
            "caption_path": str(caption_path),
            "valid": len(all_issues) == 0,
            "has_warnings": len(all_warnings) > 0,
            "issues": all_issues,
            "warnings": all_warnings,
            "image_info": {
                "width": img_result.get("width"),
                "height": img_result.get("height"),
                "mode": img_result.get("mode"),
                "file_size": img_result.get("file_size")
            },
            "caption_info": {
                "length": caption_result.get("length"),
                "preview": caption_result.get("caption", "")[:100] if caption_result.get("caption") else None
            }
        }

    def validate_directory(self, directory: Path) -> Dict:
        """
        Validate entire directory

        Args:
            directory: Training data directory

        Returns:
            Directory validation results
        """
        # Find all images
        image_files = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))

        if not image_files:
            return {
                "valid": False,
                "error": "No images found in directory",
                "total_images": 0
            }

        # Validate each pair
        results = []
        for img_path in tqdm(image_files, desc=f"  Validating {directory.name}", leave=False):
            result = self.validate_training_pair(img_path)
            results.append(result)

        # Aggregate statistics
        total = len(results)
        valid = sum(1 for r in results if r["valid"])
        with_warnings = sum(1 for r in results if r["has_warnings"])
        invalid = total - valid

        # Collect all issues
        issue_types = {}
        warning_types = {}

        for result in results:
            for issue in result["issues"]:
                issue_types[issue] = issue_types.get(issue, 0) + 1
            for warning in result["warnings"]:
                warning_types[warning] = warning_types.get(warning, 0) + 1

        return {
            "valid": invalid == 0,
            "directory": str(directory),
            "total_images": total,
            "valid_pairs": valid,
            "invalid_pairs": invalid,
            "pairs_with_warnings": with_warnings,
            "validation_rate": valid / total if total > 0 else 0,
            "issue_types": issue_types,
            "warning_types": warning_types,
            "failed_pairs": [r for r in results if not r["valid"]],
            "all_results": results
        }

    def validate_kohya_structure(self, train_data_dir: Path) -> Dict:
        """
        Validate kohya_ss training directory structure

        Expected structure:
        train_data_dir/
        ├── {repeat}_{character_name}/
        │   ├── image1.png
        │   ├── image1.txt
        │   └── ...
        └── validation/
            └── {character_name}/
                ├── val_image1.png
                └── val_image1.txt

        Args:
            train_data_dir: Root training data directory

        Returns:
            Structure validation results
        """
        issues = []
        warnings = []

        if not train_data_dir.exists():
            return {
                "valid": False,
                "issues": [f"Training directory not found: {train_data_dir}"],
                "warnings": []
            }

        # Find character directories (format: {repeat}_{name})
        char_dirs = [d for d in train_data_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.') and d.name != 'validation']

        if not char_dirs:
            issues.append("No character directories found")
            return {
                "valid": False,
                "issues": issues,
                "warnings": warnings
            }

        # Check directory naming convention
        valid_named_dirs = []
        for char_dir in char_dirs:
            # Check if name follows {repeat}_{name} format
            parts = char_dir.name.split('_', 1)
            if len(parts) == 2 and parts[0].isdigit():
                valid_named_dirs.append(char_dir)
            else:
                warnings.append(
                    f"Directory '{char_dir.name}' doesn't follow "
                    f"kohya_ss naming convention ({{repeat}}_{{name}})"
                )

        if not valid_named_dirs:
            issues.append("No properly named character directories found")

        # Check validation directory
        val_dir = train_data_dir / "validation"
        has_validation = val_dir.exists() and val_dir.is_dir()

        if not has_validation:
            warnings.append("No validation directory found (recommended)")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "character_dirs": [str(d) for d in valid_named_dirs],
            "has_validation": has_validation,
            "total_dirs": len(valid_named_dirs)
        }


def validate_training_data(
    train_data_dir: Path,
    output_report: Path = None,
    min_resolution: int = 512,
    max_aspect_ratio: float = 3.0
):
    """
    Validate complete training dataset

    Args:
        train_data_dir: Root training data directory
        output_report: Optional path for validation report
        min_resolution: Minimum image dimension
        max_aspect_ratio: Maximum aspect ratio
    """
    print(f"\n{'='*80}")
    print("YOKAI LORA TRAINING DATA VALIDATION")
    print(f"{'='*80}\n")

    print(f"📂 Training data directory: {train_data_dir}")
    print(f"📏 Min resolution: {min_resolution}px")
    print(f"📐 Max aspect ratio: {max_aspect_ratio}")
    print()

    # Create validator
    validator = TrainingDataValidator(
        min_resolution=min_resolution,
        max_aspect_ratio=max_aspect_ratio
    )

    # Step 1: Validate directory structure
    print("Step 1: Validating directory structure...")
    structure_result = validator.validate_kohya_structure(train_data_dir)

    if not structure_result["valid"]:
        print(f"❌ Structure validation failed:")
        for issue in structure_result["issues"]:
            print(f"   - {issue}")
        return

    print(f"✓ Found {structure_result['total_dirs']} character directories")

    if structure_result["warnings"]:
        print(f"⚠️  Warnings:")
        for warning in structure_result["warnings"]:
            print(f"   - {warning}")
    print()

    # Step 2: Validate each character directory
    print("Step 2: Validating character directories...")

    char_results = []
    for char_dir_str in tqdm(structure_result["character_dirs"], desc="Validating directories"):
        char_dir = Path(char_dir_str)
        result = validator.validate_directory(char_dir)
        char_results.append(result)

    # Step 3: Validate validation set if exists
    val_results = []
    if structure_result["has_validation"]:
        print("\nStep 3: Validating validation set...")
        val_dir = train_data_dir / "validation"
        val_char_dirs = [d for d in val_dir.iterdir() if d.is_dir()]

        for val_char_dir in tqdm(val_char_dirs, desc="Validating validation set"):
            result = validator.validate_directory(val_char_dir)
            val_results.append(result)

    # Generate summary
    total_train_images = sum(r["total_images"] for r in char_results)
    total_valid_train = sum(r["valid_pairs"] for r in char_results)
    total_invalid_train = sum(r["invalid_pairs"] for r in char_results)
    total_train_warnings = sum(r["pairs_with_warnings"] for r in char_results)

    total_val_images = sum(r["total_images"] for r in val_results)
    total_valid_val = sum(r["valid_pairs"] for r in val_results)
    total_invalid_val = sum(r["invalid_pairs"] for r in val_results)

    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Training Set:")
    print(f"   Total images: {total_train_images}")
    print(f"   Valid pairs: {total_valid_train} ({total_valid_train/total_train_images*100:.1f}%)")
    print(f"   Invalid pairs: {total_invalid_train}")
    print(f"   Pairs with warnings: {total_train_warnings}")

    if val_results:
        print(f"\n📊 Validation Set:")
        print(f"   Total images: {total_val_images}")
        print(f"   Valid pairs: {total_valid_val} ({total_valid_val/total_val_images*100:.1f}%)")
        print(f"   Invalid pairs: {total_invalid_val}")

    # Show common issues
    all_issues = {}
    all_warnings = {}

    for result in char_results + val_results:
        for issue, count in result["issue_types"].items():
            all_issues[issue] = all_issues.get(issue, 0) + count
        for warning, count in result["warning_types"].items():
            all_warnings[warning] = all_warnings.get(warning, 0) + count

    if all_issues:
        print(f"\n⚠️  Common Issues:")
        for issue, count in sorted(all_issues.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {issue}: {count} occurrences")

    if all_warnings:
        print(f"\n⚠️  Common Warnings:")
        for warning, count in sorted(all_warnings.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {warning}: {count} occurrences")

    # Overall assessment
    overall_valid = total_invalid_train == 0 and total_invalid_val == 0

    print(f"\n{'='*80}")
    if overall_valid:
        print("✅ VALIDATION PASSED - Training data is ready")
    else:
        print("❌ VALIDATION FAILED - Please fix issues before training")
    print(f"{'='*80}\n")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "train_data_dir": str(train_data_dir),
        "structure": structure_result,
        "training_set": {
            "total_images": total_train_images,
            "valid_pairs": total_valid_train,
            "invalid_pairs": total_invalid_train,
            "pairs_with_warnings": total_train_warnings,
            "directories": char_results
        },
        "validation_set": {
            "total_images": total_val_images,
            "valid_pairs": total_valid_val,
            "invalid_pairs": total_invalid_val,
            "directories": val_results
        } if val_results else None,
        "common_issues": all_issues,
        "common_warnings": all_warnings,
        "overall_valid": overall_valid
    }

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Validation report saved: {output_report}\n")

    # Show problematic characters
    problematic_chars = [r for r in char_results if r["invalid_pairs"] > 0]
    if problematic_chars:
        print(f"Characters with issues:")
        for result in sorted(problematic_chars, key=lambda x: x["invalid_pairs"], reverse=True)[:10]:
            char_name = Path(result["directory"]).name
            print(f"  - {char_name}: {result['invalid_pairs']}/{result['total_images']} invalid")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Yokai LoRA training data"
    )

    parser.add_argument(
        "train_data_dir",
        type=Path,
        help="Training data directory (kohya_ss format)"
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path for validation report JSON"
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=512,
        help="Minimum image dimension (default: 512)"
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=3.0,
        help="Maximum aspect ratio (default: 3.0)"
    )

    args = parser.parse_args()

    if not args.train_data_dir.exists():
        print(f"❌ Training data directory not found: {args.train_data_dir}")
        return

    validate_training_data(
        train_data_dir=args.train_data_dir,
        output_report=args.output_report,
        min_resolution=args.min_resolution,
        max_aspect_ratio=args.max_aspect_ratio
    )


if __name__ == "__main__":
    main()
