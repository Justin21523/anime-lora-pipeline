#!/usr/bin/env python3
"""
Action Sequence Extractor for AnimateDiff Training

Extracts continuous action sequences from anime episodes:
- Appearance animations (yokai entrance, summon, transformation)
- Attack sequences (special moves, combat actions)
- Motion patterns (running, jumping, flying)
- Flexible sequence lengths (8/16/32/64 frames)

Organized for AnimateDiff motion LoRA training.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import shutil
import warnings

warnings.filterwarnings("ignore")


class ActionSequenceExtractor:
    """Extracts action sequences for motion training"""

    def __init__(
        self,
        min_motion: float = 10.0,
        max_motion: float = 150.0,
        min_consistency: float = 0.7,
        sequence_lengths: List[int] = None,
    ):
        """
        Initialize extractor

        Args:
            min_motion: Minimum motion threshold
            max_motion: Maximum motion threshold (filter camera pans)
            min_consistency: Minimum consistency score (0-1)
            sequence_lengths: Target sequence lengths (default: [8, 16, 32, 64])
        """
        self.min_motion = min_motion
        self.max_motion = max_motion
        self.min_consistency = min_consistency
        self.sequence_lengths = sequence_lengths or [8, 16, 32, 64]

    def calculate_frame_similarity(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two frames

        Args:
            frame1: First frame (RGB)
            frame2: Second frame (RGB)

        Returns:
            Similarity score (0-1)
        """
        # Resize for faster processing
        small1 = cv2.resize(frame1, (64, 64))
        small2 = cv2.resize(frame2, (64, 64))

        # Calculate histogram correlation
        hist1 = cv2.calcHist(
            [small1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        hist2 = cv2.calcHist(
            [small2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )

        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        return float(max(0, similarity))

    def calculate_optical_flow(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate optical flow between frames

        Args:
            prev_frame: Previous frame (RGB)
            curr_frame: Current frame (RGB)

        Returns:
            Tuple of (motion_magnitude, flow_visualization)
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

        # Calculate flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )

        # Calculate magnitude
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        avg_magnitude = magnitude.mean()

        return float(avg_magnitude), flow

    def detect_scene_change(
        self, frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.3
    ) -> bool:
        """
        Detect if there's a scene change between frames

        Args:
            frame1: First frame
            frame2: Second frame
            threshold: Similarity threshold

        Returns:
            True if scene changed
        """
        similarity = self.calculate_frame_similarity(frame1, frame2)
        return similarity < threshold

    def analyze_sequence_consistency(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze consistency of a frame sequence

        Args:
            frames: List of frames

        Returns:
            Consistency analysis
        """
        if len(frames) < 2:
            return {"consistent": False, "score": 0.0}

        similarities = []
        motions = []

        for i in range(len(frames) - 1):
            # Check similarity (should be high - same scene)
            similarity = self.calculate_frame_similarity(frames[i], frames[i + 1])
            similarities.append(similarity)

            # Check motion (should be moderate - visible action)
            motion, _ = self.calculate_optical_flow(frames[i], frames[i + 1])
            motions.append(motion)

        avg_similarity = np.mean(similarities)
        avg_motion = np.mean(motions)
        motion_variance = np.var(motions)

        # Consistency criteria:
        # 1. High inter-frame similarity (same scene)
        # 2. Moderate motion (visible action)
        # 3. Consistent motion (not erratic)

        consistent = (
            avg_similarity >= self.min_consistency
            and self.min_motion <= avg_motion <= self.max_motion
            and motion_variance < 1000  # Not too erratic
        )

        return {
            "consistent": consistent,
            "score": float(avg_similarity),
            "avg_motion": float(avg_motion),
            "motion_variance": float(motion_variance),
            "min_similarity": float(min(similarities)),
            "max_motion": float(max(motions)),
        }

    def classify_action_type(self, frames: List[np.ndarray]) -> str:
        """
        Classify the type of action in sequence

        Args:
            frames: List of frames

        Returns:
            Action type classification
        """
        if len(frames) < 2:
            return "unknown"

        # Analyze motion patterns
        motions = []
        for i in range(len(frames) - 1):
            motion, flow = self.calculate_optical_flow(frames[i], frames[i + 1])
            motions.append(motion)

        avg_motion = np.mean(motions)
        motion_trend = np.polyfit(range(len(motions)), motions, 1)[0]

        # Analyze brightness changes (flashes, effects)
        brightness_changes = []
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            change = abs(gray2.mean() - gray1.mean())
            brightness_changes.append(change)

        avg_brightness_change = np.mean(brightness_changes)

        # Classification heuristics
        if avg_brightness_change > 30:
            return "special_effect"  # Flash/explosion effects
        elif avg_motion > 80:
            if motion_trend > 0:
                return "acceleration"  # Speed up (dash, charge)
            else:
                return "high_action"  # Fast combat/movement
        elif avg_motion > 40:
            if motion_trend > 0:
                return "entrance"  # Appearing/summoning
            else:
                return "normal_action"  # Regular movement
        elif avg_motion > 15:
            return "idle_motion"  # Standing with slight movement
        else:
            return "static"  # Minimal motion

    def extract_sequences(
        self, video_path: Path, output_dir: Path, target_lengths: List[int] = None
    ) -> List[Dict]:
        """
        Extract action sequences from video

        Args:
            video_path: Path to video file
            output_dir: Output directory
            target_lengths: Target sequence lengths (default: self.sequence_lengths)

        Returns:
            List of extracted sequences
        """
        if target_lengths is None:
            target_lengths = self.sequence_lengths

        print(f"\n📹 Extracting sequences: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        sequences = []
        frame_buffer = []
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc="  Processing frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Check for scene change
            if frame_buffer and self.detect_scene_change(frame_buffer[-1], frame_rgb):
                # Scene changed - process buffer if valid
                if len(frame_buffer) >= min(target_lengths):
                    self._process_frame_buffer(
                        frame_buffer,
                        sequences,
                        frame_idx - len(frame_buffer),
                        target_lengths,
                    )

                frame_buffer = []

            frame_buffer.append(frame_rgb)

            # Process buffer when it reaches max length
            if len(frame_buffer) >= max(target_lengths) * 2:
                # Try to extract sequences
                self._process_frame_buffer(
                    frame_buffer,
                    sequences,
                    frame_idx - len(frame_buffer),
                    target_lengths,
                )

                # Keep last few frames for continuity check
                frame_buffer = frame_buffer[-8:]

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # Process remaining buffer
        if len(frame_buffer) >= min(target_lengths):
            self._process_frame_buffer(
                frame_buffer, sequences, frame_idx - len(frame_buffer), target_lengths
            )

        print(f"✓ Extracted {len(sequences)} sequences")

        return sequences

    def _process_frame_buffer(
        self,
        buffer: List[np.ndarray],
        sequences: List[Dict],
        start_idx: int,
        target_lengths: List[int],
    ):
        """
        Process frame buffer and extract valid sequences

        Args:
            buffer: Frame buffer
            sequences: List to append valid sequences
            start_idx: Starting frame index
            target_lengths: Target sequence lengths
        """
        # Try each target length
        for length in sorted(target_lengths, reverse=True):
            if len(buffer) < length:
                continue

            # Sliding window to find best segment
            best_score = -1
            best_start = 0

            for i in range(len(buffer) - length + 1):
                segment = buffer[i : i + length]
                analysis = self.analyze_sequence_consistency(segment)

                if analysis["consistent"] and analysis["score"] > best_score:
                    best_score = analysis["score"]
                    best_start = i

            # Extract if valid
            if best_score >= self.min_consistency:
                segment = buffer[best_start : best_start + length]
                action_type = self.classify_action_type(segment)

                sequences.append(
                    {
                        "start_frame": start_idx + best_start,
                        "end_frame": start_idx + best_start + length - 1,
                        "length": length,
                        "consistency_score": best_score,
                        "action_type": action_type,
                        "frames": segment,
                    }
                )

                # Remove processed frames from buffer
                return

    def save_sequence(
        self,
        sequence: Dict,
        output_dir: Path,
        video_name: str,
        format: str = "animatediff",
    ) -> Path:
        """
        Save sequence to disk

        Args:
            sequence: Sequence data
            output_dir: Output directory
            video_name: Source video name
            format: Output format ("animatediff", "individual", "gif")

        Returns:
            Path to saved sequence
        """
        # Create sequence directory
        seq_name = (
            f"{video_name}_"
            f"seq{sequence['start_frame']:06d}_"
            f"len{sequence['length']}_"
            f"{sequence['action_type']}"
        )

        seq_dir = output_dir / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        if format == "animatediff":
            # Save frames in AnimateDiff format (numbered 0000.png, 0001.png, etc.)
            for i, frame in enumerate(sequence["frames"]):
                img = Image.fromarray(frame)
                img_path = seq_dir / f"{i:04d}.png"
                img.save(img_path)

        elif format == "individual":
            # Save with descriptive names
            for i, frame in enumerate(sequence["frames"]):
                img = Image.fromarray(frame)
                img_path = seq_dir / f"frame_{sequence['start_frame'] + i:06d}.png"
                img.save(img_path)

        elif format == "gif":
            # Save as animated GIF
            images = [Image.fromarray(frame) for frame in sequence["frames"]]
            gif_path = seq_dir.parent / f"{seq_name}.gif"
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=42,  # ~24fps
                loop=0,
            )

        # Save metadata
        metadata = {
            "start_frame": sequence["start_frame"],
            "end_frame": sequence["end_frame"],
            "length": sequence["length"],
            "action_type": sequence["action_type"],
            "consistency_score": sequence["consistency_score"],
            "source_video": video_name,
        }

        metadata_path = seq_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return seq_dir


def extract_all_sequences(
    episodes_dir: Path,
    output_dir: Path,
    sequence_lengths: List[int] = None,
    output_format: str = "animatediff",
    min_sequences_per_video: int = 5,
):
    """
    Extract action sequences from all episodes

    Args:
        episodes_dir: Directory containing episodes
        output_dir: Output directory
        sequence_lengths: Target sequence lengths
        output_format: Output format
        min_sequences_per_video: Minimum sequences to extract per video
    """
    print(f"\n{'='*80}")
    print("ACTION SEQUENCE EXTRACTION FOR ANIMATEDIFF")
    print(f"{'='*80}\n")

    print(f"Episodes directory: {episodes_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence lengths: {sequence_lengths or [8, 16, 32, 64]}")
    print(f"Output format: {output_format}")
    print()

    # Create extractor
    extractor = ActionSequenceExtractor(sequence_lengths=sequence_lengths)

    # Find all video files
    video_files = []
    for ext in ["*.mp4", "*.mkv", "*.avi", "*.ts"]:
        video_files.extend(episodes_dir.rglob(ext))

    video_files = sorted(video_files)

    all_sequences = []
    total_extracted = 0

    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = video_path.stem

        # Extract sequences
        sequences = extractor.extract_sequences(
            video_path, output_dir, sequence_lengths
        )

        # Save sequences
        for sequence in sequences:
            seq_dir = extractor.save_sequence(
                sequence, output_dir, video_name, output_format
            )

            sequence["output_dir"] = str(seq_dir)
            all_sequences.append(sequence)
            total_extracted += 1

    # Group by action type
    by_action = {}
    for seq in all_sequences:
        action = seq["action_type"]
        if action not in by_action:
            by_action[action] = []
        by_action[action].append(seq)

    # Group by length
    by_length = {}
    for seq in all_sequences:
        length = seq["length"]
        if length not in by_length:
            by_length[length] = []
        by_length[length].append(seq)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_sequences": len(all_sequences),
        "total_videos": len(video_files),
        "by_action_type": {k: len(v) for k, v in by_action.items()},
        "by_length": {k: len(v) for k, v in by_length.items()},
        "sequences": all_sequences,
    }

    metadata_path = output_dir / "sequences_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"  Total sequences: {len(all_sequences)}")
    print(f"  Total videos processed: {len(video_files)}")
    print()
    print("By action type:")
    for action, seqs in sorted(
        by_action.items(), key=lambda x: len(x[1]), reverse=True
    ):
        print(f"  {action}: {len(seqs)} sequences")
    print()
    print("By length:")
    for length, seqs in sorted(by_length.items()):
        print(f"  {length} frames: {len(seqs)} sequences")
    print()
    print(f"  Output: {output_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract action sequences for AnimateDiff training"
    )

    parser.add_argument(
        "episodes_dir", type=Path, help="Directory containing episode videos"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for sequences"
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64],
        help="Target sequence lengths (default: 8 16 32 64)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="animatediff",
        choices=["animatediff", "individual", "gif"],
        help="Output format (default: animatediff)",
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=10.0,
        help="Minimum motion threshold (default: 10.0)",
    )
    parser.add_argument(
        "--max-motion",
        type=float,
        default=150.0,
        help="Maximum motion threshold (default: 150.0)",
    )

    args = parser.parse_args()

    if not args.episodes_dir.exists():
        print(f"❌ Episodes directory not found: {args.episodes_dir}")
        return

    extract_all_sequences(
        episodes_dir=args.episodes_dir,
        output_dir=args.output_dir,
        sequence_lengths=args.lengths,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
