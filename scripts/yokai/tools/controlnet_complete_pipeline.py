#!/usr/bin/env python3
"""
Complete ControlNet Pipeline for Yokai Watch

Generates all ControlNet conditioning images in one pass:
- OpenPose: Character pose detection
- Depth: Depth map generation (anime-optimized)
- Canny: Edge detection
- Lineart: Line art extraction
- Segmentation: Character segmentation masks

Outputs training-ready dataset with all control types paired with source images.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import json
from datetime import datetime
from tqdm import tqdm
import shutil
import warnings
warnings.filterwarnings('ignore')

try:
    from controlnet_aux import OpenposeDetector, LineartDetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    print("⚠️  controlnet_aux not available, some features disabled")


class ControlNetPipeline:
    """Complete ControlNet preprocessing pipeline"""

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Initialize processors
        if CONTROLNET_AUX_AVAILABLE:
            print("🔧 Loading ControlNet processors...")
            try:
                self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                print("  ✓ OpenPose loaded")
            except:
                self.openpose = None
                print("  ⚠️  OpenPose failed to load")

            try:
                self.lineart = LineartDetector.from_pretrained("lllyasviel/ControlNet")
                print("  ✓ Lineart loaded")
            except:
                self.lineart = None
                print("  ⚠️  Lineart failed to load")
        else:
            self.openpose = None
            self.lineart = None

    def generate_canny(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """Generate Canny edge map"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # Convert to 3-channel
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges_rgb

    def generate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Generate anime-style depth map

        Simple heuristic-based depth for anime images:
        - Darker = further
        - Brighter = closer
        - Character regions = closest
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Invert (darker becomes further)
        depth = 255 - gray

        # Apply bilateral filter for edge-preserving smoothing
        depth = cv2.bilateralFilter(depth, 9, 75, 75)

        # Normalize
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to 3-channel
        depth_rgb = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        return depth_rgb

    def generate_depth_from_layers(
        self,
        character_layer: np.ndarray,
        background_layer: np.ndarray
    ) -> np.ndarray:
        """
        Generate depth from separate character and background layers

        Args:
            character_layer: RGBA character layer
            background_layer: RGB background layer

        Returns:
            Depth map (3-channel RGB)
        """
        height, width = character_layer.shape[:2]

        # Initialize depth map
        depth = np.zeros((height, width), dtype=np.float32)

        # Background depth (furthest, 0.0-0.5)
        bg_gray = cv2.cvtColor(background_layer, cv2.COLOR_RGB2GRAY)
        bg_depth = (255 - bg_gray) / 255.0 * 0.5

        depth = bg_depth

        # Character depth (closest, 0.7-1.0)
        if character_layer.shape[2] == 4:
            char_mask = character_layer[:, :, 3] > 128
            char_rgb = character_layer[:, :, :3]
            char_gray = cv2.cvtColor(char_rgb, cv2.COLOR_RGB2GRAY)
            char_depth = (255 - char_gray) / 255.0 * 0.3 + 0.7

            depth[char_mask] = char_depth[char_mask]

        # Convert to 0-255
        depth_uint8 = (depth * 255).astype(np.uint8)

        # Apply edge-preserving filter
        depth_uint8 = cv2.bilateralFilter(depth_uint8, 9, 75, 75)

        # Convert to RGB
        depth_rgb = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB)

        return depth_rgb

    def generate_openpose(self, image: Image.Image) -> Optional[np.ndarray]:
        """Generate OpenPose skeleton"""
        if self.openpose is None:
            return None

        try:
            pose = self.openpose(image)
            return np.array(pose)
        except Exception as e:
            print(f"⚠️  OpenPose failed: {e}")
            return None

    def generate_lineart(self, image: Image.Image) -> Optional[np.ndarray]:
        """Generate line art"""
        if self.lineart is None:
            # Fallback: simple edge detection
            img_array = np.array(image)
            return self.generate_canny(img_array, 100, 200)

        try:
            lineart = self.lineart(image)
            return np.array(lineart)
        except Exception as e:
            print(f"⚠️  Lineart failed: {e}")
            # Fallback
            img_array = np.array(image)
            return self.generate_canny(img_array, 100, 200)

    def generate_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Generate segmentation mask

        For RGBA images with alpha channel, use alpha as mask.
        For RGB images, create simple foreground/background mask.
        """
        if image.shape[2] == 4:
            # Use alpha channel
            alpha = image[:, :, 3]
            mask = (alpha > 128).astype(np.uint8) * 255
        else:
            # Simple foreground/background using GrabCut
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Define rectangle for probable foreground
            height, width = image.shape[:2]
            rect = (width//10, height//10, width*8//10, height*8//10)

            try:
                cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
            except:
                # Fallback: simple threshold
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert to 3-channel
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        return mask_rgb

    def process_image(
        self,
        image_path: Path,
        control_types: List[str],
        background_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Process single image to generate all control maps

        Args:
            image_path: Path to source image
            control_types: List of control types to generate
            background_path: Optional path to background image (for better depth)

        Returns:
            Dict of {control_type: control_image}
        """
        # Load image
        img = Image.open(image_path)

        # Convert to RGB/RGBA
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')

        img_array = np.array(img)

        results = {}

        # Generate each control type
        for control_type in control_types:
            if control_type == 'canny':
                results['canny'] = self.generate_canny(img_array)

            elif control_type == 'depth':
                if background_path and background_path.exists():
                    # Use layered depth
                    bg_img = np.array(Image.open(background_path).convert('RGB'))
                    results['depth'] = self.generate_depth_from_layers(img_array, bg_img)
                else:
                    # Simple depth
                    results['depth'] = self.generate_depth(img_array)

            elif control_type == 'openpose':
                pose = self.generate_openpose(img)
                if pose is not None:
                    results['openpose'] = pose

            elif control_type == 'lineart':
                lineart = self.generate_lineart(img)
                if lineart is not None:
                    results['lineart'] = lineart

            elif control_type == 'segmentation':
                results['segmentation'] = self.generate_segmentation(img_array)

        return results

    def process_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        control_types: List[str],
        background_dir: Optional[Path] = None
    ) -> Dict:
        """
        Process entire dataset

        Args:
            input_dir: Input directory with images
            output_dir: Output directory
            control_types: Control types to generate
            background_dir: Optional background directory

        Returns:
            Processing statistics
        """
        print(f"\n🎮 ControlNet Pipeline")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Control types: {', '.join(control_types)}")
        print()

        # Find images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(input_dir.glob(ext))

        if not image_files:
            return {'success': False, 'error': 'No images found'}

        # Create output structure
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        for control_type in control_types:
            (output_dir / control_type).mkdir(parents=True, exist_ok=True)

        captions_dir = output_dir / "captions"
        captions_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'by_control_type': {ct: 0 for ct in control_types}
        }

        # Process each image
        for img_path in tqdm(image_files, desc="  Processing images"):
            try:
                # Find background if available
                bg_path = None
                if background_dir:
                    bg_name = img_path.stem.replace('_character', '_background') + '.jpg'
                    bg_path = background_dir / bg_name

                # Generate controls
                controls = self.process_image(img_path, control_types, bg_path)

                if not controls:
                    stats['failed'] += 1
                    continue

                # Copy source image
                source_output = source_dir / img_path.name
                shutil.copy2(img_path, source_output)

                # Save control images
                for control_type, control_img in controls.items():
                    control_output = output_dir / control_type / img_path.name
                    Image.fromarray(control_img).save(control_output)
                    stats['by_control_type'][control_type] += 1

                # Copy caption if exists
                caption_file = img_path.with_suffix('.txt')
                if caption_file.exists():
                    caption_output = captions_dir / caption_file.name
                    shutil.copy2(caption_file, caption_output)

                stats['processed'] += 1

            except Exception as e:
                print(f"⚠️  Failed to process {img_path.name}: {e}")
                stats['failed'] += 1
                continue

        return {'success': True, 'stats': stats}


def process_controlnet_dataset(
    input_dir: Path,
    output_dir: Path,
    control_types: List[str] = None,
    background_dir: Path = None,
    device: str = "cuda"
):
    """
    Generate ControlNet training dataset

    Args:
        input_dir: Input directory with images
        output_dir: Output directory
        control_types: Control types to generate (default: all)
        background_dir: Optional background directory
        device: Processing device
    """
    print(f"\n{'='*80}")
    print("CONTROLNET DATASET GENERATION")
    print(f"{'='*80}\n")

    if control_types is None:
        control_types = ['canny', 'depth', 'openpose', 'lineart', 'segmentation']

    # Initialize pipeline
    pipeline = ControlNetPipeline(device=device)

    # Process
    result = pipeline.process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        control_types=control_types,
        background_dir=background_dir
    )

    if not result['success']:
        print(f"❌ {result.get('error', 'Processing failed')}")
        return

    stats = result['stats']

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print()
    print("Control types generated:")
    for control_type, count in stats['by_control_type'].items():
        print(f"  {control_type}: {count}")
    print()
    print(f"  Output: {output_dir}")
    print(f"{'='*80}\n")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'control_types': control_types,
        'stats': stats
    }

    metadata_path = output_dir / "controlnet_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Metadata saved: {metadata_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ControlNet training dataset"
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
        "--control-types",
        nargs="+",
        default=None,
        choices=['canny', 'depth', 'openpose', 'lineart', 'segmentation'],
        help="Control types to generate (default: all)"
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=None,
        help="Background directory for better depth maps"
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

    process_controlnet_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        control_types=args.control_types,
        background_dir=args.background_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
