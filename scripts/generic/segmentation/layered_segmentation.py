#!/usr/bin/env python3
"""
Layered Segmentation Tool for Anime Frames
Separates frames into character layer, background layer, and optional effect layer

Based on research from video processing and generation analysis:
- Character layer: Used for character LoRA training
- Background layer: Used for scene generation
- Effect layer: Used for special effects generation

Recommended models:
- Character: U²-Net / anime-segmentation (fast, accurate for 2D anime)
- Background: LaMa inpainting / SD inpainting
- Effects: CLIPSeg / SAM (optional, text-driven)
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import Tuple, Optional, Dict
import cv2
from tqdm import tqdm
import json
from datetime import datetime


class AnimeSegmentationModel:
    """Wrapper for anime-specific segmentation models"""

    def __init__(self, model_type: str = "u2net", device: str = "cuda"):
        """
        Initialize segmentation model

        Args:
            model_type: "u2net", "u2netp", "isnet-anime" (via rembg)
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model_type = model_type

        print(f"🔧 Initializing {model_type} segmentation model...")

        # Use rembg (simpler, more reliable)
        from rembg import new_session
        self.session = new_session(model_type)
        print(f"✓ Rembg model '{model_type}' loaded successfully")

    def segment(self, image: Image.Image) -> np.ndarray:
        """
        Segment character from image

        Args:
            image: PIL Image

        Returns:
            Binary mask (0-255) where 255 = character
        """
        from rembg import remove

        # Remove background using rembg
        output = remove(image, session=self.session, only_mask=True)

        # Convert to numpy array
        mask = np.array(output)

        return mask


class BackgroundInpainter:
    """Inpaint background after character removal"""

    def __init__(self, method: str = "lama", device: str = "cuda"):
        """
        Initialize inpainting method

        Args:
            method: "lama" (deep learning) or "telea" (traditional CV)
            device: "cuda" or "cpu"
        """
        self.method = method
        self.device = device

        print(f"🔧 Initializing {method} inpainting...")

        if method == "lama":
            self._init_lama()
        elif method == "telea":
            print("✓ Using OpenCV Telea algorithm (fast, traditional)")
        else:
            raise ValueError(f"Unknown inpainting method: {method}")

    def _init_lama(self):
        """Initialize LaMa model"""
        try:
            # Try to use simple-lama-inpainting
            from simple_lama_inpainting import SimpleLama
            self.lama_model = SimpleLama()
            print("✓ LaMa model loaded")
        except ImportError:
            print("⚠️  LaMa not available, falling back to OpenCV Telea")
            print("   For better quality: pip install simple-lama-inpainting")
            self.method = "telea"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint masked region

        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W) where 255 = inpaint this region

        Returns:
            Inpainted image (H, W, 3)
        """
        if self.method == "lama" and hasattr(self, 'lama_model'):
            # LaMa inpainting
            result = self.lama_model(image, mask)
            return result
        else:
            # OpenCV Telea (fast, reasonable quality)
            result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return result


class LayeredSegmentation:
    """Main class for layered segmentation pipeline"""

    def __init__(
        self,
        seg_model: str = "u2net",
        inpaint_method: str = "telea",
        device: str = "cuda"
    ):
        """
        Initialize layered segmentation pipeline

        Args:
            seg_model: Segmentation model ("u2net" or "anime-segmentation")
            inpaint_method: Inpainting method ("lama" or "telea")
            device: Device to use ("cuda" or "cpu")
        """
        self.device = device

        # Initialize models
        self.segmenter = AnimeSegmentationModel(seg_model, device)
        self.inpainter = BackgroundInpainter(inpaint_method, device)

        print("✓ Layered segmentation pipeline initialized\n")

    def process_frame(
        self,
        image_path: Path,
        output_dir: Path,
        save_layers: bool = True,
        refine_edges: bool = True
    ) -> Dict[str, Path]:
        """
        Process single frame into layers

        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            save_layers: Whether to save individual layers
            refine_edges: Apply edge refinement to mask

        Returns:
            Dictionary of output paths
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)

        # 1. Segment character
        mask = self.segmenter.segment(image)

        # 2. Refine edges (optional)
        if refine_edges:
            mask = self._refine_mask(mask)

        # 3. Extract character layer (with alpha channel)
        character_layer = self._extract_character(img_array, mask)

        # 4. Inpaint background
        background_layer = self.inpainter.inpaint(img_array, mask)

        # 5. Save outputs
        outputs = {}
        if save_layers:
            stem = image_path.stem

            # Save character layer (RGBA)
            char_path = output_dir / "character" / f"{stem}_character.png"
            char_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(character_layer, mode="RGBA").save(char_path)
            outputs['character'] = char_path

            # Save background layer (RGB)
            bg_path = output_dir / "background" / f"{stem}_background.jpg"
            bg_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(background_layer, mode="RGB").save(bg_path, quality=95)
            outputs['background'] = bg_path

            # Save mask
            mask_path = output_dir / "masks" / f"{stem}_mask.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(mask_path)
            outputs['mask'] = mask_path

        return outputs

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine mask edges using morphological operations

        Args:
            mask: Binary mask (0-255)

        Returns:
            Refined mask
        """
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Smooth edges slightly
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Re-threshold
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask

    def _extract_character(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract character with alpha channel

        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)

        Returns:
            RGBA image (H, W, 4)
        """
        # Create RGBA image
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = mask  # Alpha channel

        return rgba


def process_directory(
    input_dir: Path,
    output_dir: Path,
    seg_model: str = "u2net",
    inpaint_method: str = "telea",
    device: str = "cuda",
    pattern: str = "*.jpg"
):
    """
    Process all frames in a directory

    Args:
        input_dir: Input directory with frames
        output_dir: Output directory for layers
        seg_model: Segmentation model
        inpaint_method: Inpainting method
        device: Device to use
        pattern: File pattern to match
    """
    print(f"\n{'='*80}")
    print("LAYERED SEGMENTATION - BATCH PROCESSING")
    print(f"{'='*80}\n")

    # Find all images
    image_files = sorted(input_dir.glob(pattern))
    if not image_files:
        # Try PNG
        image_files = sorted(input_dir.glob("*.png"))

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}\n")

    # Initialize pipeline
    pipeline = LayeredSegmentation(seg_model, inpaint_method, device)

    # Process each image
    results = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_frames": len(image_files),
        "config": {
            "seg_model": seg_model,
            "inpaint_method": inpaint_method,
            "device": device
        },
        "processed_frames": []
    }

    print("🎨 Processing frames...\n")
    for img_path in tqdm(image_files, desc="Segmenting"):
        try:
            outputs = pipeline.process_frame(img_path, output_dir)
            results["processed_frames"].append({
                "input": str(img_path.name),
                "outputs": {k: str(v.name) for k, v in outputs.items()}
            })
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path.name}: {e}")
            results["processed_frames"].append({
                "input": str(img_path.name),
                "error": str(e)
            })

    # Save results
    results_path = output_dir / "segmentation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Segmentation complete!")
    print(f"  Character layers: {output_dir / 'character'}")
    print(f"  Background layers: {output_dir / 'background'}")
    print(f"  Masks: {output_dir / 'masks'}")
    print(f"  Results: {results_path}")
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Layered Segmentation for Anime Frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single episode with U²-Net and fast inpainting
  python layered_segmentation.py episode_001 --output-dir layers/episode_001

  # Process with high-quality LaMa inpainting
  python layered_segmentation.py episode_001 --inpaint lama

  # Use anime-segmentation model (more accurate)
  python layered_segmentation.py episode_001 --model anime-segmentation

  # Process on CPU
  python layered_segmentation.py episode_001 --device cpu
"""
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory with frames to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: {input_dir}_layers)"
    )
    parser.add_argument(
        "--model",
        choices=["u2net", "u2netp", "isnet-anime", "isnet-general-use"],
        default="u2net",
        help="Segmentation model (default: u2net, fast and good for anime)"
    )
    parser.add_argument(
        "--inpaint",
        choices=["telea", "lama"],
        default="telea",
        help="Inpainting method (default: telea, fast)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="File pattern to match (default: *.jpg)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir.parent / f"{args.input_dir.name}_layers"

    # Check input directory exists
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    # Process directory
    process_directory(
        args.input_dir,
        args.output_dir,
        args.model,
        args.inpaint,
        args.device,
        args.pattern
    )


if __name__ == "__main__":
    main()
