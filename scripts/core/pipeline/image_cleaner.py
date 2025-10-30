"""
Image Cleaner for deduplication and quality filtering
"""

import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import imagehash
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.logger import get_logger, create_progress_bar, print_success, print_error, print_section, print_info, print_warning
from core.utils.image_utils import check_image_quality, get_image_hash
from core.utils.path_utils import ensure_dir, list_images, safe_copy, save_json


logger = get_logger("ImageCleaner")


class ImageCleaner:
    """
    Clean and deduplicate image datasets
    """

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (512, 512),
        blur_threshold: float = 150.0,
        brightness_range: Tuple[float, float] = (30, 225),
        hash_size: int = 8,
        hash_threshold: int = 5  # Hamming distance for duplicate detection
    ):
        """
        Initialize ImageCleaner

        Args:
            min_resolution: Minimum (width, height)
            blur_threshold: Blur detection threshold
            brightness_range: Acceptable brightness range
            hash_size: Size of perceptual hash
            hash_threshold: Max Hamming distance to consider duplicates
        """
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.hash_size = hash_size
        self.hash_threshold = hash_threshold

    def compute_image_hashes(
        self,
        image_paths: List[Path]
    ) -> Dict[Path, str]:
        """
        Compute perceptual hashes for all images

        Args:
            image_paths: List of image file paths

        Returns:
            Dictionary mapping image paths to hash strings
        """
        hashes = {}

        print_section("Computing Image Hashes")

        with create_progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Hashing images...",
                total=len(image_paths)
            )

            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    img_hash = imagehash.phash(img, hash_size=self.hash_size)
                    hashes[img_path] = str(img_hash)
                except Exception as e:
                    logger.error(f"Error hashing {img_path.name}: {e}")

                progress.update(task, advance=1)

        print_success(f"Hashed {len(hashes)} images")
        return hashes

    def find_duplicates(
        self,
        image_hashes: Dict[Path, str]
    ) -> List[List[Path]]:
        """
        Find duplicate images based on perceptual hash similarity

        Args:
            image_hashes: Dictionary of image paths to hashes

        Returns:
            List of duplicate groups (each group is a list of paths)
        """
        print_section("Finding Duplicates")

        # Convert hash strings to imagehash objects
        hash_objects = {
            path: imagehash.hex_to_hash(hash_str)
            for path, hash_str in image_hashes.items()
        }

        # Find duplicates
        duplicate_groups = []
        processed = set()

        with create_progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Comparing images...",
                total=len(hash_objects)
            )

            for path1, hash1 in hash_objects.items():
                if path1 in processed:
                    progress.update(task, advance=1)
                    continue

                # Find similar images
                group = [path1]
                for path2, hash2 in hash_objects.items():
                    if path1 == path2 or path2 in processed:
                        continue

                    # Calculate Hamming distance
                    distance = hash1 - hash2

                    if distance <= self.hash_threshold:
                        group.append(path2)
                        processed.add(path2)

                if len(group) > 1:
                    duplicate_groups.append(group)

                processed.add(path1)
                progress.update(task, advance=1)

        print_info(f"Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups

    def select_best_from_duplicates(
        self,
        duplicate_group: List[Path]
    ) -> Path:
        """
        Select the best image from a duplicate group

        Args:
            duplicate_group: List of duplicate image paths

        Returns:
            Path to the best image
        """
        best_image = None
        best_score = -1

        for img_path in duplicate_group:
            try:
                # Quality check
                passes, metrics = check_image_quality(
                    img_path,
                    self.min_resolution,
                    self.blur_threshold,
                    self.brightness_range
                )

                # Score based on blur and resolution
                score = metrics.get('blur_score', 0) + (metrics.get('width', 0) * metrics.get('height', 0)) / 10000

                if score > best_score:
                    best_score = score
                    best_image = img_path

            except Exception as e:
                logger.error(f"Error evaluating {img_path.name}: {e}")

        return best_image or duplicate_group[0]

    def quality_filter(
        self,
        image_paths: List[Path]
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """
        Filter images by quality

        Args:
            image_paths: List of image paths to check

        Returns:
            Tuple of (passed_images, failed_images_with_reasons)
        """
        print_section("Quality Filtering")

        passed = []
        failed = []

        with create_progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Checking quality...",
                total=len(image_paths)
            )

            for img_path in image_paths:
                try:
                    passes, metrics = check_image_quality(
                        img_path,
                        self.min_resolution,
                        self.blur_threshold,
                        self.brightness_range
                    )

                    if passes:
                        passed.append(img_path)
                    else:
                        # Determine failure reason
                        reasons = []
                        if not metrics.get('resolution_ok', False):
                            reasons.append(f"low_res_{metrics.get('width')}x{metrics.get('height')}")
                        if metrics.get('is_blurry', False):
                            reasons.append(f"blurry_{metrics.get('blur_score', 0):.1f}")
                        if not metrics.get('brightness_ok', False):
                            reasons.append(f"brightness_{metrics.get('brightness', 0):.1f}")

                        failed.append((img_path, ", ".join(reasons)))

                except Exception as e:
                    failed.append((img_path, f"error: {str(e)}"))

                progress.update(task, advance=1)

        print_success(f"Passed: {len(passed)}, Failed: {len(failed)}")
        return passed, failed

    def clean_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        remove_duplicates: bool = True,
        remove_low_quality: bool = True,
        copy_files: bool = True,
        save_report: bool = True
    ) -> Dict[str, any]:
        """
        Clean image directory

        Args:
            input_dir: Input directory with images
            output_dir: Output directory for cleaned images
            remove_duplicates: Remove duplicate images
            remove_low_quality: Remove low quality images
            copy_files: Copy files to output (if False, only generate report)
            save_report: Save cleaning report

        Returns:
            Cleaning statistics
        """
        ensure_dir(output_dir)

        print_section(f"Cleaning Directory: {input_dir.name}")

        # Get all images
        all_images = list_images(input_dir, recursive=False)
        print_info(f"Found {len(all_images)} images")

        stats = {
            'input_images': len(all_images),
            'duplicates_removed': 0,
            'quality_failed': 0,
            'output_images': 0
        }

        working_set = all_images.copy()

        # Step 1: Remove duplicates
        if remove_duplicates and len(working_set) > 1:
            image_hashes = self.compute_image_hashes(working_set)
            duplicate_groups = self.find_duplicates(image_hashes)

            # Keep only best from each group
            images_to_remove = set()
            for group in duplicate_groups:
                best = self.select_best_from_duplicates(group)
                for img in group:
                    if img != best:
                        images_to_remove.add(img)

            working_set = [img for img in working_set if img not in images_to_remove]
            stats['duplicates_removed'] = len(images_to_remove)

            print_info(f"Removed {len(images_to_remove)} duplicates")

        # Step 2: Quality filter
        if remove_low_quality:
            passed, failed = self.quality_filter(working_set)
            working_set = passed
            stats['quality_failed'] = len(failed)

        stats['output_images'] = len(working_set)

        # Step 3: Copy cleaned images
        if copy_files:
            print_section("Copying Cleaned Images")

            with create_progress_bar() as progress:
                task = progress.add_task(
                    "[cyan]Copying files...",
                    total=len(working_set)
                )

                for img_path in working_set:
                    dst_path = output_dir / img_path.name
                    safe_copy(img_path, dst_path, overwrite=True)
                    progress.update(task, advance=1)

            print_success(f"Copied {len(working_set)} images to {output_dir}")

        # Save report
        if save_report:
            report = {
                'statistics': stats,
                'cleaned_images': [str(img) for img in working_set]
            }

            report_path = output_dir / "cleaning_report.json"
            save_json(report, report_path)
            print_info(f"Report saved to: {report_path}")

        # Summary
        print_section("Cleaning Summary")
        print_info(f"Input images: {stats['input_images']}")
        print_info(f"Duplicates removed: {stats['duplicates_removed']}")
        print_info(f"Quality failed: {stats['quality_failed']}")
        print_success(f"Output images: {stats['output_images']}")

        return stats


def main():
    """Test ImageCleaner"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean and deduplicate images")
    parser.add_argument("input_dir", type=Path, help="Input directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--no-quality", action="store_true", help="Skip quality filtering")

    args = parser.parse_args()

    cleaner = ImageCleaner()

    stats = cleaner.clean_directory(
        input_dir=args.input_dir,
        output_dir=args.output,
        remove_duplicates=not args.no_dedup,
        remove_low_quality=not args.no_quality,
        copy_files=True,
        save_report=True
    )

    print(f"\n✓ Cleaned {stats['output_images']} images")


if __name__ == "__main__":
    main()
