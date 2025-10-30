#!/usr/bin/env python3
"""
YOLO-Based Character Detection (Parallel)
Directly detect and extract complete characters from frames
NO segmentation needed - captures full characters including all yokai types

Advantages over segmentation:
- ✅ Complete characters (not just heads/faces)
- ✅ Faster (skip segmentation step)
- ✅ Works for all character types (humans + all yokai forms)
- ✅ Parallel processing for maximum speed
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
import cv2
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def process_single_frame(args: Tuple[Path, Path, str, float, int]) -> Dict:
    """
    Process single frame with YOLO detection

    Args:
        args: (frame_path, output_dir, model_path, confidence_threshold, min_size)

    Returns:
        Detection results
    """
    frame_path, output_dir, model_path, conf_threshold, min_size = args

    try:
        # Load YOLO model (per process)
        model = YOLO(model_path)

        # Load image
        image = cv2.imread(str(frame_path))
        if image is None:
            return {"status": "error", "frame": str(frame_path.name), "error": "Failed to load image"}

        h, w = image.shape[:2]

        # Detect objects
        results = model(image, conf=conf_threshold, verbose=False)[0]

        # Extract character crops
        detections = []
        for idx, box in enumerate(results.boxes):
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Filter by size
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < min_size or bbox_h < min_size:
                continue

            # Crop character
            character_crop = image[y1:y2, x1:x2]

            # Save crop
            stem = frame_path.stem
            crop_filename = f"{stem}_char{idx:02d}.png"
            crop_path = output_dir / crop_filename

            # Convert BGR to RGB for PIL
            character_rgb = cv2.cvtColor(character_crop, cv2.COLOR_BGR2RGB)
            Image.fromarray(character_rgb).save(crop_path)

            detections.append({
                "crop_path": str(crop_path),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "class": int(box.cls[0])
            })

        return {
            "status": "success",
            "frame": str(frame_path.name),
            "num_detections": len(detections),
            "detections": detections
        }

    except Exception as e:
        return {
            "status": "error",
            "frame": str(frame_path.name),
            "error": str(e)
        }


def detect_characters_parallel(
    input_dir: Path,
    output_dir: Path,
    model_path: str = "yolov8x.pt",
    confidence_threshold: float = 0.3,
    min_size: int = 50,
    num_workers: int = None,
    pattern: str = "*.jpg"
):
    """
    Detect and extract characters from all frames in parallel

    Args:
        input_dir: Input directory with frames
        output_dir: Output directory for character crops
        model_path: YOLO model path
        confidence_threshold: Detection confidence threshold
        min_size: Minimum character size in pixels
        num_workers: Number of parallel workers
        pattern: File pattern to match
    """
    print(f"\n{'='*80}")
    print("YOLO-BASED PARALLEL CHARACTER DETECTION")
    print(f"{'='*80}\n")

    if not YOLO_AVAILABLE:
        print("❌ ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return

    # Find all frames
    image_files = []
    if input_dir.is_dir():
        # Single directory
        image_files = sorted(input_dir.glob(pattern))
        if not image_files:
            image_files = sorted(input_dir.glob("*.png"))
    else:
        # Assume it's a file with list of paths
        with open(input_dir) as f:
            image_files = [Path(line.strip()) for line in f if line.strip()]

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    # Determine workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)

    print(f"📊 Configuration:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Frames: {len(image_files)}")
    print(f"  Model: {model_path}")
    print(f"  Confidence: {confidence_threshold}")
    print(f"  Min Size: {min_size}px")
    print(f"  Workers: {num_workers}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments
    args_list = [
        (img_path, output_dir, model_path, confidence_threshold, min_size)
        for img_path in image_files
    ]

    # Process in parallel
    print(f"\n🚀 Detecting characters with {num_workers} workers...\n")

    start_time = datetime.now()
    results = []

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_frame, args_list),
            total=len(args_list),
            desc="Detecting",
            unit="frame"
        ):
            results.append(result)

    elapsed = (datetime.now() - start_time).total_seconds()

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    total_detections = sum(r.get("num_detections", 0) for r in results if r["status"] == "success")

    print(f"\n{'='*80}")
    print(f"✓ Detection Complete!")
    print(f"{'='*80}")
    print(f"  Frames processed: {len(results)}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Total characters detected: {total_detections}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {len(results)/elapsed:.2f} frames/sec")
    print(f"\n  Character crops: {output_dir}/")
    print(f"{'='*80}\n")

    # Save results JSON
    results_file = output_dir / "detection_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_frames": len(results),
            "success": success,
            "failed": failed,
            "total_characters_detected": total_detections,
            "elapsed_seconds": elapsed,
            "frames_per_second": len(results) / elapsed,
            "model": model_path,
            "confidence_threshold": confidence_threshold,
            "results": results
        }, f, indent=2)

    print(f"Results saved to: {results_file}")


def detect_from_multiple_dirs(
    input_dirs: List[Path],
    output_dir: Path,
    **kwargs
):
    """
    Detect characters from multiple episode directories

    Args:
        input_dirs: List of episode directories
        output_dir: Output directory for all character crops
        **kwargs: Additional arguments for detect_characters_parallel
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for episode_dir in input_dirs:
        print(f"\n{'#'*80}")
        print(f"Processing: {episode_dir.name}")
        print(f"{'#'*80}")

        episode_output = output_dir / episode_dir.name
        episode_output.mkdir(parents=True, exist_ok=True)

        detect_characters_parallel(
            input_dir=episode_dir,
            output_dir=episode_output,
            **kwargs
        )


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Based Parallel Character Detection"
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with frames or episode directories"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for character crops"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x.pt",
        help="YOLO model (yolov8n/s/m/l/x)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=50,
        help="Minimum character size in pixels"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jpg",
        help="File pattern"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively"
    )

    args = parser.parse_args()

    if args.recursive and args.input_dir.is_dir():
        # Process all subdirectories
        subdirs = [d for d in args.input_dir.iterdir() if d.is_dir()]
        print(f"📁 Found {len(subdirs)} subdirectories")

        detect_from_multiple_dirs(
            input_dirs=subdirs,
            output_dir=args.output_dir,
            model_path=args.model,
            confidence_threshold=args.confidence,
            min_size=args.min_size,
            num_workers=args.num_workers,
            pattern=args.pattern
        )
    else:
        # Single directory
        detect_characters_parallel(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model,
            confidence_threshold=args.confidence,
            min_size=args.min_size,
            num_workers=args.num_workers,
            pattern=args.pattern
        )


if __name__ == "__main__":
    main()
