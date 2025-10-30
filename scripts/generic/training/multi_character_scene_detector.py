#!/usr/bin/env python3
"""
Multi-Character Scene Detector
Detects and extracts frames with multiple characters for interaction training
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import json
import shutil
from PIL import Image


class MultiCharacterSceneDetector:
    """Detect scenes with multiple characters"""

    def __init__(
        self,
        layered_frames_dir: Path,
        output_dir: Path,
        min_characters: int = 2,
        max_characters: int = 5,
        min_character_size: int = 5000,  # pixels
        use_hard_links: bool = True
    ):
        """
        Initialize multi-character detector

        Args:
            layered_frames_dir: Directory with layered frames
            output_dir: Output directory for multi-character scenes
            min_characters: Minimum number of characters
            max_characters: Maximum number of characters
            min_character_size: Minimum character size in pixels
            use_hard_links: Use hard links instead of copying
        """
        self.layered_frames_dir = Path(layered_frames_dir)
        self.output_dir = Path(output_dir)
        self.min_characters = min_characters
        self.max_characters = max_characters
        self.min_character_size = min_character_size
        self.use_hard_links = use_hard_links

        print(f"👥 Multi-Character Scene Detector")
        print(f"  Input: {layered_frames_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Character Range: {min_characters}-{max_characters}")
        print(f"  Min Character Size: {min_character_size} pixels")
        print(f"  Link Mode: {'Hard Links' if use_hard_links else 'Copy'}")

    def detect_characters_in_image(self, image: np.ndarray) -> List[Dict]:
        """
        Detect separate characters in an image using connected components

        Args:
            image: Input image (RGBA)

        Returns:
            List of detected character regions
        """
        if image.shape[2] != 4:
            # No alpha channel, can't detect characters reliably
            return []

        alpha = image[:, :, 3]

        # Threshold alpha to get binary mask
        _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        characters = []

        # Skip label 0 (background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by size
            if area < self.min_character_size:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            characters.append({
                'id': i,
                'area': int(area),
                'bbox': [int(x), int(y), int(w), int(h)],
                'centroid': [float(cx), float(cy)]
            })

        return characters

    def analyze_spatial_layout(self, characters: List[Dict]) -> Dict:
        """
        Analyze spatial layout of characters

        Args:
            characters: List of character detections

        Returns:
            Layout analysis
        """
        if len(characters) < 2:
            return {'layout_type': 'single'}

        # Sort by x-coordinate (left to right)
        sorted_chars = sorted(characters, key=lambda c: c['centroid'][0])

        # Calculate distances
        distances = []
        for i in range(len(sorted_chars) - 1):
            x1 = sorted_chars[i]['centroid'][0]
            x2 = sorted_chars[i + 1]['centroid'][0]
            distances.append(x2 - x1)

        avg_distance = np.mean(distances) if distances else 0

        # Determine layout
        if len(characters) == 2:
            layout_type = 'pair'
        elif len(characters) == 3:
            layout_type = 'trio'
        else:
            layout_type = 'group'

        # Check if characters are in a line (similar y-coordinates)
        y_coords = [c['centroid'][1] for c in characters]
        y_variance = np.var(y_coords)

        if y_variance < 1000:  # Low variance = horizontal line
            arrangement = 'horizontal'
        else:
            arrangement = 'scattered'

        return {
            'layout_type': layout_type,
            'arrangement': arrangement,
            'num_characters': len(characters),
            'avg_spacing': float(avg_distance),
            'y_variance': float(y_variance)
        }

    def detect_multi_character_scenes(self) -> Dict:
        """
        Detect all multi-character scenes

        Returns:
            Dictionary with detection results
        """
        print(f"\n🔍 Detecting multi-character scenes...")

        # Find all episodes
        episodes = sorted([d for d in self.layered_frames_dir.iterdir()
                          if d.is_dir() and d.name.startswith('episode_')])

        if not episodes:
            print(f"  ⚠️  No episode directories found")
            return {}

        print(f"  Found {len(episodes)} episodes")

        multi_char_scenes = []
        total_frames = 0

        for episode_dir in tqdm(episodes, desc="Processing episodes"):
            char_dir = episode_dir / "character"

            if not char_dir.exists():
                continue

            char_files = list(char_dir.glob("*.png"))
            total_frames += len(char_files)

            for char_file in char_files:
                # Read image
                image = cv2.imread(str(char_file), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue

                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                # Detect characters
                characters = self.detect_characters_in_image(image)

                # Filter by character count
                if self.min_characters <= len(characters) <= self.max_characters:
                    layout = self.analyze_spatial_layout(characters)

                    scene_info = {
                        'character_path': str(char_file),
                        'episode': episode_dir.name,
                        'filename': char_file.name,
                        'num_characters': len(characters),
                        'characters': characters,
                        'layout': layout
                    }

                    # Check for corresponding background
                    bg_name = char_file.name.replace('_character.png', '_background.jpg')
                    bg_path = episode_dir / "background" / bg_name
                    if bg_path.exists():
                        scene_info['background_path'] = str(bg_path)

                    multi_char_scenes.append(scene_info)

        return {
            'total_frames': total_frames,
            'multi_char_frames': len(multi_char_scenes),
            'scenes': multi_char_scenes
        }

    def organize_scenes(self):
        """
        Organize multi-character scenes
        """
        # Detect scenes
        detection = self.detect_multi_character_scenes()

        if not detection['multi_char_frames']:
            print(f"  ⚠️  No multi-character scenes found")
            return

        print(f"\n📊 Detection Results:")
        print(f"  Total frames processed: {detection['total_frames']}")
        print(f"  Multi-character frames: {detection['multi_char_frames']}")
        print(f"  Detection rate: {100 * detection['multi_char_frames'] / detection['total_frames']:.1f}%")

        # Organize by character count
        by_count = {}
        for scene in detection['scenes']:
            count = scene['num_characters']
            if count not in by_count:
                by_count[count] = []
            by_count[count].append(scene)

        print(f"\n  Distribution by character count:")
        for count in sorted(by_count.keys()):
            print(f"    {count} characters: {len(by_count[count])} frames")

        # Copy/link files to organized structure
        for count, scenes in tqdm(by_count.items(), desc="Organizing"):
            count_dir = self.output_dir / f"{count}_characters"
            char_subdir = count_dir / "character"
            bg_subdir = count_dir / "background"
            char_subdir.mkdir(parents=True, exist_ok=True)
            bg_subdir.mkdir(parents=True, exist_ok=True)

            for scene in scenes:
                # Copy character image
                char_src = Path(scene['character_path'])
                char_dst = char_subdir / f"{scene['episode']}_{scene['filename']}"

                if self.use_hard_links and 'ai_warehouse' in str(self.output_dir):
                    try:
                        if not char_dst.exists():
                            char_dst.hardlink_to(char_src)
                    except Exception:
                        if not char_dst.exists():
                            shutil.copy2(char_src, char_dst)
                else:
                    if not char_dst.exists():
                        shutil.copy2(char_src, char_dst)

                # Copy background if exists
                if 'background_path' in scene:
                    bg_src = Path(scene['background_path'])
                    bg_dst = bg_subdir / f"{scene['episode']}_{bg_src.name}"

                    if self.use_hard_links and 'ai_warehouse' in str(self.output_dir):
                        try:
                            if not bg_dst.exists():
                                bg_dst.hardlink_to(bg_src)
                        except Exception:
                            if not bg_dst.exists():
                                shutil.copy2(bg_src, bg_dst)
                    else:
                        if not bg_dst.exists():
                            shutil.copy2(bg_src, bg_dst)

        # Save metadata
        metadata = {
            'total_frames': detection['total_frames'],
            'multi_char_frames': detection['multi_char_frames'],
            'detection_rate': detection['multi_char_frames'] / detection['total_frames'],
            'by_count': {str(k): len(v) for k, v in by_count.items()},
            'min_characters': self.min_characters,
            'max_characters': self.max_characters,
            'min_character_size': self.min_character_size
        }

        metadata_path = self.output_dir / "multi_character_detection.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save detailed scene info
        detailed_path = self.output_dir / "scene_details.json"
        with open(detailed_path, 'w') as f:
            json.dump(detection, f, indent=2)

        print(f"\n✅ Multi-character scene organization complete!")
        print(f"  Total detected: {detection['multi_char_frames']}")
        print(f"  Output: {self.output_dir}")

        print(f"\n📊 Dataset Structure:")
        print(f"  {self.output_dir}/")
        for count in sorted(by_count.keys()):
            print(f"    ├── {count}_characters/  ({len(by_count[count])} scenes)")
            print(f"    │   ├── character/")
            print(f"    │   └── background/")
        print(f"    ├── multi_character_detection.json")
        print(f"    └── scene_details.json")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Detect and organize multi-character scenes"
    )
    parser.add_argument(
        "layered_frames_dir",
        type=Path,
        help="Directory with layered frames"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for multi-character scenes"
    )
    parser.add_argument(
        "--min-characters",
        type=int,
        default=2,
        help="Minimum number of characters (default: 2)"
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=5,
        help="Maximum number of characters (default: 5)"
    )
    parser.add_argument(
        "--min-character-size",
        type=int,
        default=5000,
        help="Minimum character size in pixels (default: 5000)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of using hard links"
    )

    args = parser.parse_args()

    detector = MultiCharacterSceneDetector(
        layered_frames_dir=args.layered_frames_dir,
        output_dir=args.output_dir,
        min_characters=args.min_characters,
        max_characters=args.max_characters,
        min_character_size=args.min_character_size,
        use_hard_links=not args.copy
    )

    detector.organize_scenes()


if __name__ == "__main__":
    main()
