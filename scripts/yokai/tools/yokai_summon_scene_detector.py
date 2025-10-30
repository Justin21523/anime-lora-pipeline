#!/usr/bin/env python3
"""
Yokai Summon Scene Detector

Detects special yokai summon sequences with spectacular effects:
- Visual detection: Special effects, light beams, magic circles, energy waves
- Audio integration: Sound effects and music analysis
- Key frame extraction: Capture the most spectacular moments
- Scene classification: First summon, battle summon, evolution, etc.

These special scenes are perfect for training effect-focused LoRAs.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  librosa not available, audio detection disabled")


class SummonSceneDetector:
    """Detects yokai summon scenes using visual and audio cues"""

    def __init__(
        self,
        brightness_threshold: float = 180.0,
        saturation_threshold: float = 100.0,
        motion_threshold: float = 50.0,
        audio_threshold: float = 0.7,
        min_duration: int = 15,  # frames
        max_duration: int = 120  # frames
    ):
        """
        Initialize detector

        Args:
            brightness_threshold: Brightness threshold for effect detection
            saturation_threshold: Saturation threshold for colorful effects
            motion_threshold: Motion magnitude threshold
            audio_threshold: Audio energy threshold
            min_duration: Minimum scene duration in frames
            max_duration: Maximum scene duration in frames
        """
        self.brightness_threshold = brightness_threshold
        self.saturation_threshold = saturation_threshold
        self.motion_threshold = motion_threshold
        self.audio_threshold = audio_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration

    def analyze_frame_effects(self, frame: np.ndarray) -> Dict:
        """
        Analyze visual effects in a frame

        Args:
            frame: RGB frame

        Returns:
            Effect analysis results
        """
        # Convert to HSV for better effect detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Analyze brightness (high brightness = flash effects)
        brightness = hsv[:, :, 2].mean()
        max_brightness = hsv[:, :, 2].max()

        # Analyze saturation (high saturation = colorful effects)
        saturation = hsv[:, :, 1].mean()

        # Detect circular patterns (magic circles)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=20,
            maxRadius=200
        )

        num_circles = len(circles[0]) if circles is not None else 0

        # Detect radial patterns (light beams)
        edges = cv2.Canny(gray, 50, 150)
        radial_score = self._detect_radial_pattern(edges)

        # Detect color bursts (multicolor effects)
        color_variety = self._calculate_color_variety(frame)

        return {
            'brightness': float(brightness),
            'max_brightness': float(max_brightness),
            'saturation': float(saturation),
            'num_circles': int(num_circles),
            'radial_score': float(radial_score),
            'color_variety': float(color_variety),
            'has_flash': max_brightness > 250 and brightness > self.brightness_threshold,
            'has_effects': saturation > self.saturation_threshold and color_variety > 0.5
        }

    def _detect_radial_pattern(self, edges: np.ndarray) -> float:
        """
        Detect radial light beam patterns

        Args:
            edges: Edge map

        Returns:
            Radial pattern score (0-1)
        """
        height, width = edges.shape
        center_y, center_x = height // 2, width // 2

        # Create radial mask (check edges emanating from center)
        angles = np.linspace(0, 2 * np.pi, 36)  # 36 directions
        radial_score = 0

        for angle in angles:
            dx = int(np.cos(angle) * width // 4)
            dy = int(np.sin(angle) * height // 4)

            x = center_x + dx
            y = center_y + dy

            if 0 <= x < width and 0 <= y < height:
                # Check if there's an edge along this radial direction
                if edges[y, x] > 0:
                    radial_score += 1

        return radial_score / len(angles)

    def _calculate_color_variety(self, frame: np.ndarray) -> float:
        """
        Calculate color variety (multiple distinct colors = effects)

        Args:
            frame: RGB frame

        Returns:
            Color variety score (0-1)
        """
        # Resize for faster processing
        small = cv2.resize(frame, (64, 64))

        # Convert to HSV and quantize hue
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]

        # Count distinct hues (ignore very dark/light pixels)
        mask = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
        hues = hue[mask]

        if len(hues) == 0:
            return 0.0

        # Quantize to 12 bins (30 degrees each)
        bins = np.histogram(hues, bins=12, range=(0, 180))[0]
        non_zero_bins = np.sum(bins > 0)

        return non_zero_bins / 12.0

    def calculate_optical_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> float:
        """
        Calculate motion magnitude using optical flow

        Args:
            prev_frame: Previous frame (RGB)
            curr_frame: Current frame (RGB)

        Returns:
            Average motion magnitude
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

        # Calculate optical flow
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
            flags=0
        )

        # Calculate magnitude
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        avg_magnitude = magnitude.mean()

        return float(avg_magnitude)

    def analyze_audio(
        self,
        audio_path: Path,
        start_time: float,
        duration: float
    ) -> Dict:
        """
        Analyze audio segment for summon cues

        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds

        Returns:
            Audio analysis results
        """
        if not AUDIO_AVAILABLE:
            return {'audio_available': False}

        try:
            # Load audio segment
            y, sr = librosa.load(
                audio_path,
                sr=22050,
                offset=start_time,
                duration=duration
            )

            # Energy analysis
            rms = librosa.feature.rms(y=y)[0]
            avg_energy = float(rms.mean())
            max_energy = float(rms.max())

            # Spectral features (detect special effect sounds)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_centroid = float(spectral_centroid.mean())

            # Zero crossing rate (detect high-frequency effects)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = float(zcr.mean())

            # Detect sudden changes (summon sound effects)
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            num_onsets = len(librosa.onset.onset_detect(
                onset_envelope=onset_strength,
                sr=sr,
                units='time'
            ))

            return {
                'audio_available': True,
                'avg_energy': avg_energy,
                'max_energy': max_energy,
                'avg_spectral_centroid': avg_centroid,
                'avg_zcr': avg_zcr,
                'num_onsets': int(num_onsets),
                'has_sound_effects': max_energy > self.audio_threshold and num_onsets > 3
            }

        except Exception as e:
            print(f"⚠️  Audio analysis failed: {e}")
            return {'audio_available': False, 'error': str(e)}

    def detect_summon_scenes(
        self,
        video_path: Path,
        audio_path: Path = None,
        fps: float = 23.976
    ) -> List[Dict]:
        """
        Detect all summon scenes in a video

        Args:
            video_path: Path to video file
            audio_path: Optional path to audio file
            fps: Video frame rate

        Returns:
            List of detected summon scenes
        """
        print(f"\n🔍 Detecting summon scenes: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        scenes = []
        current_scene = None
        prev_frame = None

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="  Analyzing frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze visual effects
            effects = self.analyze_frame_effects(frame_rgb)

            # Calculate motion
            motion = 0.0
            if prev_frame is not None:
                motion = self.calculate_optical_flow(prev_frame, frame_rgb)

            # Check if this frame has summon indicators
            is_summon_frame = (
                effects['has_flash'] or
                effects['has_effects'] or
                effects['num_circles'] > 0 or
                effects['radial_score'] > 0.3 or
                motion > self.motion_threshold
            )

            if is_summon_frame:
                if current_scene is None:
                    # Start new scene
                    current_scene = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'peak_frame': frame_idx,
                        'peak_brightness': effects['max_brightness'],
                        'effects': [effects],
                        'motions': [motion]
                    }
                else:
                    # Extend current scene
                    current_scene['end_frame'] = frame_idx
                    current_scene['effects'].append(effects)
                    current_scene['motions'].append(motion)

                    # Update peak frame
                    if effects['max_brightness'] > current_scene['peak_brightness']:
                        current_scene['peak_frame'] = frame_idx
                        current_scene['peak_brightness'] = effects['max_brightness']
            else:
                # End current scene if exists
                if current_scene is not None:
                    duration = current_scene['end_frame'] - current_scene['start_frame']

                    # Only keep scenes within duration range
                    if self.min_duration <= duration <= self.max_duration:
                        # Analyze audio if available
                        if audio_path and AUDIO_AVAILABLE:
                            start_time = current_scene['start_frame'] / fps
                            duration_sec = duration / fps
                            current_scene['audio'] = self.analyze_audio(
                                audio_path,
                                start_time,
                                duration_sec
                            )

                        # Calculate scene score
                        current_scene['score'] = self._calculate_scene_score(current_scene)

                        scenes.append(current_scene)

                    current_scene = None

            prev_frame = frame_rgb
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # Close final scene if exists
        if current_scene is not None:
            duration = current_scene['end_frame'] - current_scene['start_frame']
            if self.min_duration <= duration <= self.max_duration:
                if audio_path and AUDIO_AVAILABLE:
                    start_time = current_scene['start_frame'] / fps
                    duration_sec = duration / fps
                    current_scene['audio'] = self.analyze_audio(
                        audio_path,
                        start_time,
                        duration_sec
                    )

                current_scene['score'] = self._calculate_scene_score(current_scene)
                scenes.append(current_scene)

        print(f"✓ Detected {len(scenes)} potential summon scenes")

        # Sort by score (highest first)
        scenes.sort(key=lambda x: x['score'], reverse=True)

        return scenes

    def _calculate_scene_score(self, scene: Dict) -> float:
        """
        Calculate overall scene score

        Args:
            scene: Scene data

        Returns:
            Scene score (0-100)
        """
        score = 0.0

        # Visual score (0-60)
        effects = scene['effects']
        avg_brightness = np.mean([e['brightness'] for e in effects])
        avg_saturation = np.mean([e['saturation'] for e in effects])
        max_circles = max([e['num_circles'] for e in effects])
        avg_radial = np.mean([e['radial_score'] for e in effects])
        avg_color_variety = np.mean([e['color_variety'] for e in effects])

        visual_score = (
            (avg_brightness / 255.0) * 15 +
            (avg_saturation / 255.0) * 15 +
            min(max_circles / 3.0, 1.0) * 10 +
            avg_radial * 10 +
            avg_color_variety * 10
        )

        score += visual_score

        # Motion score (0-20)
        motions = scene['motions']
        avg_motion = np.mean(motions)
        motion_score = min(avg_motion / 100.0, 1.0) * 20

        score += motion_score

        # Audio score (0-20)
        if 'audio' in scene and scene['audio'].get('audio_available', False):
            audio = scene['audio']
            if audio.get('has_sound_effects', False):
                audio_score = 20
            else:
                audio_score = (audio.get('max_energy', 0) / 2.0) * 20

            score += audio_score

        return float(score)

    def extract_scene_frames(
        self,
        video_path: Path,
        scene: Dict,
        output_dir: Path,
        extract_mode: str = "key"
    ) -> List[Path]:
        """
        Extract frames from detected scene

        Args:
            video_path: Path to video file
            scene: Scene data
            output_dir: Output directory
            extract_mode: "key" (peak frame only), "all" (all frames), "sample" (sample frames)

        Returns:
            List of extracted frame paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))

        extracted = []

        if extract_mode == "key":
            # Extract only peak frame
            frames_to_extract = [scene['peak_frame']]
        elif extract_mode == "all":
            # Extract all frames
            frames_to_extract = range(scene['start_frame'], scene['end_frame'] + 1)
        else:  # sample
            # Extract sample frames
            duration = scene['end_frame'] - scene['start_frame']
            sample_interval = max(1, duration // 8)  # ~8 frames
            frames_to_extract = range(
                scene['start_frame'],
                scene['end_frame'] + 1,
                sample_interval
            )

        for frame_idx in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Save frame
                output_name = f"scene_{scene['start_frame']:06d}_frame_{frame_idx:06d}.png"
                output_path = output_dir / output_name
                img.save(output_path)

                extracted.append(output_path)

        cap.release()

        return extracted


def detect_and_extract_summons(
    episodes_dir: Path,
    output_dir: Path,
    extract_mode: str = "key",
    min_score: float = 50.0,
    use_audio: bool = True
):
    """
    Detect and extract summon scenes from all episodes

    Args:
        episodes_dir: Directory containing episode folders
        output_dir: Output directory
        extract_mode: Frame extraction mode
        min_score: Minimum scene score to extract
        use_audio: Use audio analysis
    """
    print(f"\n{'='*80}")
    print("YOKAI SUMMON SCENE DETECTION")
    print(f"{'='*80}\n")

    print(f"Episodes directory: {episodes_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Extract mode: {extract_mode}")
    print(f"Min score: {min_score}")
    print(f"Audio analysis: {use_audio and AUDIO_AVAILABLE}")
    print()

    # Create detector
    detector = SummonSceneDetector()

    # Find all episodes
    episode_dirs = sorted([d for d in episodes_dir.iterdir() if d.is_dir()])

    all_scenes = []

    for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
        # Find video file (assume mp4 or mkv)
        video_files = list(episode_dir.glob("*.mp4")) + list(episode_dir.glob("*.mkv"))

        if not video_files:
            continue

        video_path = video_files[0]

        # Find audio file if using audio
        audio_path = None
        if use_audio and AUDIO_AVAILABLE:
            audio_files = list(episode_dir.glob("*.wav")) + list(episode_dir.glob("*.mp3"))
            if audio_files:
                audio_path = audio_files[0]

        # Detect scenes
        scenes = detector.detect_summon_scenes(video_path, audio_path)

        # Filter by score and extract
        for scene in scenes:
            if scene['score'] >= min_score:
                # Extract frames
                scene_output_dir = output_dir / episode_dir.name / f"scene_{scene['start_frame']:06d}"
                extracted = detector.extract_scene_frames(
                    video_path,
                    scene,
                    scene_output_dir,
                    extract_mode
                )

                scene['episode'] = episode_dir.name
                scene['extracted_frames'] = [str(p) for p in extracted]
                all_scenes.append(scene)

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_scenes': len(all_scenes),
        'extract_mode': extract_mode,
        'min_score': min_score,
        'scenes': all_scenes
    }

    metadata_path = output_dir / "summon_scenes_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("DETECTION COMPLETE")
    print(f"{'='*80}")
    print(f"  Total scenes detected: {len(all_scenes)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Detect yokai summon scenes with special effects"
    )

    parser.add_argument(
        "episodes_dir",
        type=Path,
        help="Directory containing episode folders"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for extracted scenes"
    )
    parser.add_argument(
        "--extract-mode",
        type=str,
        default="key",
        choices=["key", "all", "sample"],
        help="Frame extraction mode (default: key - peak frame only)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=50.0,
        help="Minimum scene score to extract (default: 50.0)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio analysis"
    )

    args = parser.parse_args()

    if not args.episodes_dir.exists():
        print(f"❌ Episodes directory not found: {args.episodes_dir}")
        return

    detect_and_extract_summons(
        episodes_dir=args.episodes_dir,
        output_dir=args.output_dir,
        extract_mode=args.extract_mode,
        min_score=args.min_score,
        use_audio=not args.no_audio
    )


if __name__ == "__main__":
    main()
