"""
Video Processor for extracting keyframes from anime videos
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import timedelta
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.logger import get_logger, create_progress_bar, print_success, print_error, print_section, print_info
from core.utils.image_utils import is_image_blurry, calculate_brightness, check_image_quality
from core.utils.path_utils import ensure_dir


logger = get_logger("VideoProcessor")


class VideoProcessor:
    """
    Extract keyframes from video files using scene detection
    """

    def __init__(
        self,
        min_scene_length: float = 1.0,
        threshold: float = 27.0,
        min_resolution: Tuple[int, int] = (512, 512),
        blur_threshold: float = 150.0,
        brightness_range: Tuple[float, float] = (30, 225)
    ):
        """
        Initialize VideoProcessor

        Args:
            min_scene_length: Minimum scene length in seconds
            threshold: Scene detection sensitivity (lower = more sensitive)
            min_resolution: Minimum frame resolution (width, height)
            blur_threshold: Blur detection threshold
            brightness_range: Acceptable brightness range (min, max)
        """
        self.min_scene_length = min_scene_length
        self.threshold = threshold
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range

    def detect_scenes(self, video_path: Path) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries in video

        Args:
            video_path: Path to video file

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        logger.info(f"Detecting scenes in {video_path.name}")

        try:
            # Create video manager
            video_manager = VideoManager([str(video_path)])
            scene_manager = SceneManager()

            # Add ContentDetector for scene detection
            scene_manager.add_detector(
                ContentDetector(
                    threshold=self.threshold,
                    min_scene_len=int(self.min_scene_length * video_manager.get_framerate())
                )
            )

            # Perform scene detection
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)

            # Get scene list
            scene_list = scene_manager.get_scene_list()
            video_manager.release()

            # Convert to seconds
            scenes = []
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                scenes.append((start_time, end_time))

            logger.info(f"Detected {len(scenes)} scenes")
            return scenes

        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            return []

    def extract_frame_at_time(
        self,
        video_path: Path,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame at specific timestamp

        Args:
            video_path: Path to video file
            timestamp: Time in seconds

        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            # Set position
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            # Read frame
            ret, frame = cap.read()
            cap.release()

            if ret:
                return frame
            else:
                return None

        except Exception as e:
            logger.error(f"Error extracting frame at {timestamp}s: {e}")
            return None

    def check_frame_quality(
        self,
        frame: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if frame meets quality criteria

        Args:
            frame: Frame as numpy array (BGR)

        Returns:
            Tuple of (passes_check, metrics)
        """
        metrics = {}

        try:
            # Check resolution
            h, w = frame.shape[:2]
            metrics['width'] = w
            metrics['height'] = h
            metrics['resolution_ok'] = w >= self.min_resolution[0] and h >= self.min_resolution[1]

            # Check blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['blur_score'] = float(laplacian_var)
            metrics['is_blurry'] = laplacian_var < self.blur_threshold

            # Check brightness
            brightness = np.mean(gray)
            metrics['brightness'] = float(brightness)
            metrics['brightness_ok'] = self.brightness_range[0] <= brightness <= self.brightness_range[1]

            # Overall pass
            passes = (
                metrics['resolution_ok'] and
                not metrics['is_blurry'] and
                metrics['brightness_ok']
            )

            return passes, metrics

        except Exception as e:
            logger.error(f"Error checking frame quality: {e}")
            return False, {'error': str(e)}

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        extract_first_frame_only: bool = True,
        save_metadata: bool = True
    ) -> Dict[str, any]:
        """
        Process video and extract keyframes

        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            extract_first_frame_only: Only extract first frame of each scene
            save_metadata: Save metadata JSON file

        Returns:
            Processing statistics
        """
        ensure_dir(output_dir)

        print_section(f"Processing Video: {video_path.name}")

        # Detect scenes
        scenes = self.detect_scenes(video_path)

        if not scenes:
            print_error("No scenes detected")
            return {'error': 'No scenes detected'}

        print_info(f"Found {len(scenes)} scenes")

        # Process scenes
        extracted_frames = []
        failed_frames = []
        quality_rejected = []

        with create_progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Extracting frames...",
                total=len(scenes)
            )

            for scene_idx, (start_time, end_time) in enumerate(scenes):
                # Extract first frame of scene (usually clearest)
                timestamp = start_time + 0.1  # Slightly after scene start

                # Extract frame
                frame = self.extract_frame_at_time(video_path, timestamp)

                if frame is None:
                    failed_frames.append({
                        'scene_idx': scene_idx,
                        'timestamp': timestamp,
                        'reason': 'extraction_failed'
                    })
                    progress.update(task, advance=1)
                    continue

                # Check quality
                passes_quality, metrics = self.check_frame_quality(frame)

                if not passes_quality:
                    quality_rejected.append({
                        'scene_idx': scene_idx,
                        'timestamp': timestamp,
                        'metrics': metrics
                    })
                    progress.update(task, advance=1)
                    continue

                # Save frame
                frame_filename = f"scene_{scene_idx:04d}_t{timestamp:.2f}s.png"
                frame_path = output_dir / frame_filename

                cv2.imwrite(str(frame_path), frame)

                # Record metadata
                extracted_frames.append({
                    'filename': frame_filename,
                    'scene_idx': scene_idx,
                    'timestamp': timestamp,
                    'scene_duration': end_time - start_time,
                    'quality_metrics': metrics
                })

                progress.update(task, advance=1)

        # Statistics
        stats = {
            'video_file': video_path.name,
            'total_scenes': len(scenes),
            'frames_extracted': len(extracted_frames),
            'frames_failed': len(failed_frames),
            'frames_rejected': len(quality_rejected),
            'output_dir': str(output_dir)
        }

        print_success(f"Extracted {len(extracted_frames)} frames")
        print_info(f"Failed: {len(failed_frames)}, Rejected: {len(quality_rejected)}")

        # Save metadata
        if save_metadata:
            metadata = {
                'statistics': stats,
                'extracted_frames': extracted_frames,
                'failed_frames': failed_frames,
                'quality_rejected': quality_rejected
            }

            metadata_path = output_dir / "extraction_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print_info(f"Metadata saved to: {metadata_path}")

        return stats

    def batch_process_videos(
        self,
        video_paths: List[Path],
        output_root: Path,
        create_subdirs: bool = True
    ) -> List[Dict[str, any]]:
        """
        Process multiple videos

        Args:
            video_paths: List of video file paths
            output_root: Root output directory
            create_subdirs: Create subdirectory for each video

        Returns:
            List of processing statistics for each video
        """
        ensure_dir(output_root)

        all_stats = []

        for video_path in video_paths:
            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}")
                continue

            # Determine output directory
            if create_subdirs:
                output_dir = output_root / video_path.stem
            else:
                output_dir = output_root

            # Process video
            stats = self.process_video(
                video_path=video_path,
                output_dir=output_dir,
                extract_first_frame_only=True,
                save_metadata=True
            )

            all_stats.append(stats)

        return all_stats


def main():
    """Test VideoProcessor"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract keyframes from video")
    parser.add_argument("video", type=Path, help="Video file path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=27.0, help="Scene detection threshold")
    parser.add_argument("--min-scene-length", type=float, default=1.0, help="Minimum scene length (seconds)")

    args = parser.parse_args()

    processor = VideoProcessor(
        min_scene_length=args.min_scene_length,
        threshold=args.threshold
    )

    stats = processor.process_video(
        video_path=args.video,
        output_dir=args.output
    )

    print(f"\n✓ Processed video: {stats}")


if __name__ == "__main__":
    main()
