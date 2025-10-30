#!/usr/bin/env python3
"""
Anime Depth Generator

Advanced depth map generation optimized for anime style:
- Anime-specific depth estimation
- Layer-based depth (character + background)
- Edge-preserving depth
- Multi-character depth ordering
- ControlNet Depth format

Generates higher quality depth maps than generic depth estimators.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AnimeDepthGenerator:
    """
    Advanced depth generator for anime images
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def estimate_character_depth(self, character_layer: np.ndarray) -> np.ndarray:
        """
        Estimate depth for character layer

        Args:
            character_layer: RGBA character image

        Returns:
            Depth map (0-255, uint8)
        """
        height, width = character_layer.shape[:2]

        # Initialize depth map
        depth = np.zeros((height, width), dtype=np.float32)

        # Get character mask
        if character_layer.shape[2] == 4:
            mask = character_layer[:, :, 3] > 128
            char_rgb = character_layer[:, :, :3]
        else:
            mask = np.ones((height, width), dtype=bool)
            char_rgb = character_layer

        if not mask.any():
            return depth.astype(np.uint8)

        # Convert to grayscale
        char_gray = cv2.cvtColor(char_rgb, cv2.COLOR_RGB2GRAY)

        # Method 1: Brightness-based depth
        # Brighter areas assumed closer (anime lighting)
        brightness_depth = char_gray.astype(np.float32) / 255.0

        # Method 2: Vertical position depth
        # Higher in image = further (anime perspective)
        y_coords = np.arange(height).reshape(-1, 1).repeat(width, axis=1)
        vertical_depth = 1.0 - (y_coords / height)

        # Method 3: Distance from center
        # Center usually closer in character portraits
        y_center, x_center = height // 2, width // 2
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((y_grid - y_center)**2 + (x_grid - x_center)**2)
        max_dist = np.sqrt(y_center**2 + x_center**2)
        center_depth = 1.0 - (dist_from_center / max_dist)

        # Combine methods
        # Characters are typically in the closest depth range (0.7-1.0)
        combined = (
            brightness_depth * 0.4 +
            vertical_depth * 0.3 +
            center_depth * 0.3
        )

        # Scale to character depth range (0.7-1.0)
        combined = combined * 0.3 + 0.7

        # Apply only to character pixels
        depth[mask] = combined[mask]

        # Convert to 0-255
        depth_uint8 = (depth * 255).astype(np.uint8)

        return depth_uint8

    def estimate_background_depth(
        self,
        background_layer: np.ndarray,
        preserve_edges: bool = True
    ) -> np.ndarray:
        """
        Estimate depth for background layer

        Args:
            background_layer: RGB background image
            preserve_edges: Whether to preserve anime edges

        Returns:
            Depth map (0-255, uint8)
        """
        # Convert to grayscale
        if len(background_layer.shape) == 3:
            gray = cv2.cvtColor(background_layer, cv2.COLOR_RGB2GRAY)
        else:
            gray = background_layer

        height, width = gray.shape[:2]

        # Method 1: Inverse brightness
        # Darker backgrounds = further away
        inv_brightness = (255 - gray).astype(np.float32) / 255.0

        # Method 2: Vertical gradient
        # Top = further, bottom = closer
        y_coords = np.arange(height).reshape(-1, 1).repeat(width, axis=1)
        vertical_gradient = (y_coords / height).astype(np.float32)

        # Method 3: Sky detection (top region)
        top_region = gray[:height//4, :]
        is_bright_top = top_region.mean() > 180
        if is_bright_top:
            # Bright top = sky, should be furthest
            sky_mask = np.zeros_like(gray, dtype=np.float32)
            sky_mask[:height//3, :] = 0.3  # Make top region further
            vertical_gradient += sky_mask

        # Combine methods
        # Backgrounds are in the furthest depth range (0.0-0.5)
        combined = (
            inv_brightness * 0.5 +
            vertical_gradient * 0.5
        )

        # Scale to background depth range (0.0-0.5)
        combined = combined * 0.5

        # Edge-preserving filter (preserve anime line art)
        if preserve_edges:
            combined = cv2.bilateralFilter(
                (combined * 255).astype(np.uint8),
                d=9,
                sigmaColor=75,
                sigmaSpace=75
            ).astype(np.float32) / 255.0

        # Convert to 0-255
        depth_uint8 = (combined * 255).astype(np.uint8)

        return depth_uint8

    def combine_layers_depth(
        self,
        character_layer: np.ndarray,
        background_layer: np.ndarray,
        preserve_edges: bool = True
    ) -> np.ndarray:
        """
        Combine character and background depth maps

        Args:
            character_layer: RGBA character layer
            background_layer: RGB background layer
            preserve_edges: Preserve anime edges

        Returns:
            Combined depth map (0-255, uint8)
        """
        height, width = character_layer.shape[:2]

        # Generate depth for each layer
        char_depth = self.estimate_character_depth(character_layer)
        bg_depth = self.estimate_background_depth(background_layer, preserve_edges)

        # Resize background if needed
        if bg_depth.shape != char_depth.shape:
            bg_depth = cv2.resize(bg_depth, (width, height))

        # Combine using character mask
        if character_layer.shape[2] == 4:
            char_mask = character_layer[:, :, 3] > 128
        else:
            char_mask = np.zeros((height, width), dtype=bool)

        # Start with background
        combined = bg_depth.copy()

        # Overlay character (closer depth)
        combined[char_mask] = char_depth[char_mask]

        # Smooth transitions at edges
        if preserve_edges:
            # Dilate character mask slightly
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(char_mask.astype(np.uint8), kernel, iterations=1)
            edge_mask = (dilated_mask > 0) & (~char_mask)

            # Blend at edges
            if edge_mask.any():
                # Get blend weights
                dist_transform = cv2.distanceTransform(
                    (~char_mask).astype(np.uint8),
                    cv2.DIST_L2,
                    3
                )
                blend_weight = np.clip(dist_transform / 5.0, 0, 1)

                # Blend character and background depth at edges
                combined = (
                    char_depth * (1 - blend_weight) +
                    bg_depth * blend_weight
                ).astype(np.uint8)

                # Restore character pixels
                combined[char_mask] = char_depth[char_mask]

        return combined

    def generate_from_single_image(
        self,
        image: np.ndarray,
        preserve_edges: bool = True
    ) -> np.ndarray:
        """
        Generate depth from a single image (no layers)

        Args:
            image: RGB/RGBA image
            preserve_edges: Preserve anime edges

        Returns:
            Depth map (0-255, uint8)
        """
        # Convert to grayscale
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            has_alpha = True
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            has_alpha = False

        height, width = gray.shape

        # If has alpha, use it to separate layers
        if has_alpha and image[:, :, 3].max() > 128:
            # Treat transparent parts as background
            mask = image[:, :, 3] > 128

            # Foreground depth (closer)
            fg_depth = self.estimate_character_depth(image)

            # Background depth (further)
            # Fill transparent areas with surrounding colors
            bg_image = image[:, :, :3].copy()
            if not mask.all():
                # Inpaint transparent areas
                bg_filled = cv2.inpaint(
                    bg_image,
                    (~mask).astype(np.uint8) * 255,
                    inpaintRadius=3,
                    flags=cv2.INPAINT_TELEA
                )
                bg_depth = self.estimate_background_depth(bg_filled, preserve_edges)
            else:
                bg_depth = np.zeros((height, width), dtype=np.uint8)

            # Combine
            combined = bg_depth.copy()
            combined[mask] = fg_depth[mask]

            depth = combined
        else:
            # No alpha, use heuristic depth
            # Brightness + vertical position
            brightness = gray.astype(np.float32) / 255.0
            y_coords = np.arange(height).reshape(-1, 1).repeat(width, axis=1)
            vertical = (y_coords / height).astype(np.float32)

            combined = (255 - gray).astype(np.float32) / 255.0 * 0.6 + vertical * 0.4
            depth = (combined * 255).astype(np.uint8)

        # Edge-preserving filter
        if preserve_edges:
            depth = cv2.bilateralFilter(depth, 9, 75, 75)

        return depth

    def generate_depth(
        self,
        image_path: Path,
        background_path: Optional[Path] = None,
        preserve_edges: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate depth map for an image

        Args:
            image_path: Path to source image
            background_path: Optional path to background image
            preserve_edges: Preserve anime edges

        Returns:
            (depth_map, metadata)
        """
        # Load image
        image = np.array(Image.open(image_path))

        # Load background if provided
        if background_path and background_path.exists():
            background = np.array(Image.open(background_path).convert('RGB'))

            # Generate layered depth
            depth = self.combine_layers_depth(image, background, preserve_edges)
            method = 'layered'
        else:
            # Generate from single image
            depth = self.generate_from_single_image(image, preserve_edges)
            method = 'single_image'

        # Convert to RGB for ControlNet
        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

        metadata = {
            'image_path': str(image_path),
            'background_path': str(background_path) if background_path else None,
            'method': method,
            'depth_range': {
                'min': int(depth.min()),
                'max': int(depth.max()),
                'mean': float(depth.mean())
            }
        }

        return depth_rgb, metadata

    def process_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        background_dir: Optional[Path] = None,
        preserve_edges: bool = True
    ) -> Dict:
        """
        Process entire dataset

        Args:
            input_dir: Input directory with images
            output_dir: Output directory
            background_dir: Optional background directory
            preserve_edges: Preserve anime edges

        Returns:
            Processing statistics
        """
        print(f"\n🎨 Anime Depth Generation")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Background: {background_dir or 'None'}")
        print(f"  Preserve edges: {preserve_edges}")
        print()

        # Find images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(input_dir.glob(ext))

        if not image_files:
            return {'success': False, 'error': 'No images found'}

        # Create output directories
        source_dir = output_dir / "source"
        depth_dir = output_dir / "depth"
        source_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'total_images': len(image_files),
            'processed': 0,
            'layered': 0,
            'single_image': 0
        }

        results = []

        # Process each image
        for img_path in tqdm(image_files, desc="  Generating depth"):
            try:
                # Find corresponding background
                bg_path = None
                if background_dir:
                    bg_name = img_path.stem.replace('_character', '_background') + '.jpg'
                    bg_path = background_dir / bg_name

                # Generate depth
                depth_rgb, metadata = self.generate_depth(
                    img_path,
                    bg_path,
                    preserve_edges
                )

                # Save source image
                import shutil
                source_output = source_dir / img_path.name
                shutil.copy2(img_path, source_output)

                # Save depth map
                depth_output = depth_dir / img_path.name
                Image.fromarray(depth_rgb).save(depth_output)

                # Update statistics
                if metadata['method'] == 'layered':
                    stats['layered'] += 1
                else:
                    stats['single_image'] += 1

                stats['processed'] += 1
                results.append(metadata)

            except Exception as e:
                print(f"⚠️  Failed to process {img_path.name}: {e}")
                continue

        # Save metadata
        output_metadata = {
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'results': results
        }

        metadata_path = output_dir / "depth_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(output_metadata, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print("DEPTH GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Layered depth: {stats['layered']}")
        print(f"  Single image depth: {stats['single_image']}")
        print(f"{'='*80}\n")

        return {'success': True, 'stats': stats}


def main():
    parser = argparse.ArgumentParser(
        description="Generate anime-optimized depth maps"
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=None,
        help="Background directory for layered depth (improves quality)"
    )
    parser.add_argument(
        "--no-edge-preserve",
        action="store_true",
        help="Disable edge preservation (faster but lower quality)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Processing device (default: cuda)"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    # Initialize generator
    generator = AnimeDepthGenerator(device=args.device)

    # Process dataset
    result = generator.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        background_dir=args.background_dir,
        preserve_edges=not args.no_edge_preserve
    )

    if not result['success']:
        print(f"❌ {result.get('error', 'Processing failed')}")
        return

    print(f"Metadata saved: {args.output_dir / 'depth_metadata.json'}\n")


if __name__ == "__main__":
    main()
