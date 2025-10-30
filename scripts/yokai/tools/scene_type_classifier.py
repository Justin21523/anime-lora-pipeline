#!/usr/bin/env python3
"""
Scene Type Classifier

Classifies scene types with audio-assisted analysis:
- Visual scene classification (indoor/outdoor, battle/daily, time of day)
- Audio environment detection (ambient sounds, music genre)
- Integration with background LoRA preparation
- Scene metadata enrichment

Helps organize scenes for background/environment LoRA training.
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

warnings.filterwarnings("ignore")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not available, audio analysis disabled")


class SceneTypeClassifier:
    """Classifies scene types using visual and audio features"""

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Scene type categories
        self.scene_types = {
            # 0) 世界層級：先判這個，就知道之後要去哪一套細分類
            "realm": {
                "human_town": [
                    "springdale",
                    "uptown springdale",
                    "downtown springdale",
                    "blossom heights",
                    "shopper's row",
                    "breezy hills",
                    "sakura new town",
                    "residential area",
                    "shopping street",
                ],  # human-world ordinary town :contentReference[oaicite:5]{index=5}
                "rural_human": [
                    "harrisville",
                    "old springdale",
                    "countryside",
                    "farm area",
                    "mountain village",
                ],  # rural map in YW2/YW3 :contentReference[oaicite:6]{index=6}
                "resort_coastal": [
                    "san fantastico",
                    "beach resort",
                    "seaside",
                    "port town",
                ],  # summer / beach episodes :contentReference[oaicite:7]{index=7}
                "bbq_region": [
                    "st. peanutsburg",
                    "bbq",
                    "american town",
                    "merican style street",
                ],  # YW3 american side :contentReference[oaicite:8]{index=8}
                "yo_kai_world": [
                    "yo-kai world",
                    "gera gera land",
                    "paradise springs",
                    "new yo-kai city",
                    "cluvian continent",
                    "hell's kitchen",
                    "enma palace",
                    "yo-kai world (past)",
                    "yo-kai world (future)",
                ],  # non-human realm :contentReference[oaicite:9]{index=9}
                "special_dimension": [
                    "oni time",
                    "terror time",
                    "nightmare realm",
                    "infinite inferno",
                    "hooligan road",
                    "hungry pass",
                ],  # time-limited scary spaces :contentReference[oaicite:10]{index=10}
            },
            # 1) 具體場景地點（你本來就有的 location，我幫你細到「哪一條街」級別）
            "location": {
                # --- human world towns ---
                "residential_uptown": [
                    "uptown springdale",
                    "suburban street",
                    "quiet housing",
                    "small park near houses",
                    "apartment front",
                    "residential road",
                ],  # 主角家外、鄰居家外 :contentReference[oaicite:11]{index=11}
                "downtown_commercial": [
                    "downtown springdale",
                    "shopping street",
                    "convenience store front",
                    "flower road",
                    "everymart entrance",
                    "market street",
                    "arcade street",
                ],  # 店很多、人很多那段路 :contentReference[oaicite:12]{index=12}
                "historic_hill": [
                    "blossom heights",
                    "old houses",
                    "shrine path",
                    "steps to shrine",
                    "cemetery on hill",
                ],  # 有神社、舊房子那條 :contentReference[oaicite:13]{index=13}
                "forest_mountain": [
                    "mt. wildwood",
                    "forest path",
                    "sacred tree",
                    "shrine in forest",
                    "mountain trail",
                    "river in mountain",
                ],  # 開局去抓昆蟲的那座山 :contentReference[oaicite:14]{index=14}
                "school": [
                    "springdale elementary",
                    "classroom",
                    "school corridor",
                    "schoolyard",
                    "gym",
                    "music room",
                ],  # 很多任務都在這裡 :contentReference[oaicite:15]{index=15}
                "harbor_beach": [
                    "beach",
                    "pier",
                    "fishing spot",
                    "coast road",
                    "seaside restaurant",
                    "port market",
                ],
                # --- yo-kai world style ---
                "yo_kai_city": [
                    "new yo-kai city",
                    "yokai-world street",
                    "fantasy street",
                    "floating platforms",
                    "weird shops",
                    "spirit market",
                ],  # 很妖的市街 :contentReference[oaicite:16]{index=16}
                "enma_palace": [
                    "enma palace",
                    "royal hall",
                    "throne room",
                    "demon court",
                ],  # 王族區域 :contentReference[oaicite:17]{index=17}
                "amusement_park": [
                    "gera gera land",
                    "theme park",
                    "carnival",
                    "fun rides",
                    "stage show",
                    "festival-like park",
                ],  # 妖怪遊樂園 :contentReference[oaicite:18]{index=18}
                # --- dungeons / special routes ---
                "dungeon_tower": [
                    "fuki tower",
                    "business tower dungeon",
                    "high-rise interior",
                    "office floors",
                    "elevator hall",
                ],  # 塔型地城 :contentReference[oaicite:19]{index=19}
                "dungeon_forest": [
                    "gloombell forest",
                    "deep forest dungeon",
                    "foggy woods",
                ],  # 森林型地城 :contentReference[oaicite:20]{index=20}
                "dungeon_highway": [
                    "gold gleaming highway",
                    "road-like dungeon",
                    "endless road",
                ],  # 道路型地城 :contentReference[oaicite:21]{index=21}
                "underground_waterway": [
                    "underground waterway",
                    "sewer",
                    "drainage tunnel",
                    "canal under city",
                ],  # 一代就有的水道 :contentReference[oaicite:22]{index=22}
                "museum_facility": [
                    "gourd pond museum",
                    "exhibition hall",
                    "gallery",
                    "artifact room",
                ],  # 博物館室內 echo 好認 :contentReference[oaicite:23]{index=23}
                # --- everyday indoor ---
                "player_home": [
                    "nate's house",
                    "katie's house",
                    "kid's bedroom",
                    "living room",
                    "kitchen",
                    "bathroom",
                ],
                "convenience_store": [
                    "everymart",
                    "phantomart",
                    "market counter",
                    "small shop",
                    "cashier area",
                ],  # 會有掃條碼聲/自動門聲那種 :contentReference[oaicite:24]{index=24}
                "restaurant_cafe": [
                    "ramen shop",
                    "sushi bar",
                    "oden stand",
                    "festival food stall",
                    "curry shop",
                ],
                "shrine_temple": [
                    "shinto shrine",
                    "temple yard",
                    "torii gate",
                    "offering hall",
                ],
                "transport": [
                    "train station",
                    "platform",
                    "bus stop",
                    "yokai train",
                    "elevator to yokai world",
                ],  # mirapo / 電梯通道也能塞這裡 :contentReference[oaicite:25]{index=25}
            },
            # 2) 時間維度（你原本也有）
            "time": {
                "day": "daytime",
                "night": "nighttime",
                "sunset": "sunset or sunrise",
                "festival_night": "night with lanterns / stalls / taiko",
                "indoor_lit": "indoor with artificial light",
            },
            # 3) 活動/情境（給你之後要做 multi-label 用的）
            "activity": {
                "daily_life": ["walking", "shopping", "talking at home", "school day"],
                "investigation": [
                    "searching for yo-kai",
                    "using yo-kai watch",
                    "bug catching",
                ],
                "battle": ["yo-kai battle", "boss battle", "oni chase"],
                "event_festival": [
                    "summer festival",
                    "lantern festival",
                    "dance stage",
                ],
                "stealth_escape": ["terror time escape", "avoid oni", "run in dungeon"],
                "travel": ["train travel", "yokai elevator", "mirapo warp"],
            },
            # 4) 環境型態（你說的室內/戶外要吃這裡）
            "environment": {
                "indoor_home": [
                    "interior",
                    "home interior",
                    "apartment room",
                    "living room",
                    "bedroom",
                ],
                "indoor_public": [
                    "store interior",
                    "school interior",
                    "museum interior",
                    "office floor",
                    "hospital-like",
                ],
                "outdoor_urban": [
                    "street scene",
                    "shopping street",
                    "downtown road",
                    "residential street",
                    "traffic",
                ],
                "outdoor_nature": [
                    "forest",
                    "mountain",
                    "river bank",
                    "park",
                    "shrine in forest",
                ],
                "outdoor_coastal": ["beach", "seaside", "harbor", "port town"],
                "underground": [
                    "sewer",
                    "underground waterway",
                    "cave",
                    "dungeon tunnel",
                ],
                "fantasy_space": [
                    "yo-kai world",
                    "floating stage",
                    "hell's kitchen",
                    "amusement park fantasy",
                    "demon palace",
                ],
            },
            # 5) 音訊環境（audio classifier 可以直接用這些 keyword 去匹配）
            "audio_env": {
                "quiet_residential": [
                    "cicadas",
                    "distant car",
                    "light wind",
                    "suburban ambience",
                ],
                "urban_busy": [
                    "traffic",
                    "people chatter",
                    "shop jingles",
                    "crosswalk beeps",
                ],
                "school_bell": ["school bell", "children voices", "gym echo"],
                "forest_ambient": ["birds", "insects", "stream", "footsteps on grass"],
                "shrine_ambient": [
                    "wind chime",
                    "bell strike",
                    "light footsteps on stone",
                ],
                "beach_wave": ["wave", "seagull", "port machinery"],
                "dungeon_echo": [
                    "low reverb footsteps",
                    "drip water",
                    "metal gate",
                    "wind tunnel",
                ],
                "themepark_bgm": [
                    "upbeat bgm",
                    "crowd cheer",
                    "announcement",
                    "looping attraction music",
                ],  # Gera Gera Land
                "festival_taiko": [
                    "taiko drum",
                    "festival shout",
                    "lantern night crowd",
                ],
                "oni_time_alarm": [
                    "warning siren",
                    "heavy footsteps",
                    "demon roar",
                ],  # Terror Time 特徵音 :contentReference[oaicite:26]{index=26}
            },
        }

    def analyze_visual_features(self, image: np.ndarray) -> Dict:
        """
        Analyze visual features of a scene

        Args:
            image: RGB image array

        Returns:
            Dict of visual features
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        features = {}

        # 1. Color histogram analysis
        # Warm colors (red/orange) -> indoor
        # Cool colors (blue/green) -> outdoor
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        warm_ratio = (h_hist[0:30].sum() + h_hist[150:180].sum()) / h_hist.sum()
        cool_ratio = h_hist[90:150].sum() / h_hist.sum()

        features["warm_color_ratio"] = float(warm_ratio)
        features["cool_color_ratio"] = float(cool_ratio)

        # 2. Brightness distribution
        # Bright = day, Dark = night
        brightness = hsv[:, :, 2]
        features["mean_brightness"] = float(brightness.mean())
        features["brightness_std"] = float(brightness.std())

        # Sky detection (top 1/3 of image)
        top_third = brightness[: brightness.shape[0] // 3, :]
        features["sky_brightness"] = float(top_third.mean())

        # 3. Edge density
        # High edges = urban/indoor, Low edges = nature
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features["edge_density"] = float(
            edges.sum() / (edges.shape[0] * edges.shape[1])
        )

        # 4. Saturation
        # High saturation = special effects/summon
        saturation = hsv[:, :, 1]
        features["mean_saturation"] = float(saturation.mean())
        features["saturation_std"] = float(saturation.std())

        # 5. Color temperature (using LAB)
        # a channel: green (-) to red (+)
        # Positive = warm (indoor), Negative = cool (outdoor)
        a_channel = lab[:, :, 1]
        features["color_temperature"] = float(a_channel.mean())

        # 6. Texture complexity
        # Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features["texture_complexity"] = float(laplacian.var())

        return features

    def analyze_audio_features(
        self, audio_path: Path, start_time: float = 0.0, duration: float = 2.0
    ) -> Optional[Dict]:
        """
        Analyze audio features

        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds

        Returns:
            Dict of audio features or None if failed
        """
        if not LIBROSA_AVAILABLE or not audio_path.exists():
            return None

        try:
            # Load audio segment
            y, sr = librosa.load(
                str(audio_path), offset=start_time, duration=duration, sr=22050
            )

            features = {}

            # 1. Spectral centroid
            # High = bright sounds (effects), Low = ambient
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = float(spectral_centroid.mean())

            # 2. Zero crossing rate
            # High = noisy (battle), Low = tonal (daily)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["zero_crossing_rate"] = float(zcr.mean())

            # 3. RMS energy
            # High = loud (battle), Low = quiet (daily)
            rms = librosa.feature.rms(y=y)[0]
            features["rms_energy"] = float(rms.mean())

            # 4. Tempo
            # Fast = battle, Slow = daily
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)

            # 5. Spectral rolloff
            # High = bright/harsh, Low = soft
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features["spectral_rolloff"] = float(rolloff.mean())

            return features

        except Exception as e:
            print(f"⚠️  Audio analysis failed: {e}")
            return None

    def classify_location(self, visual_features: Dict) -> Dict[str, float]:
        """
        Classify scene location based on visual features

        Returns:
            Dict of {location_type: confidence}
        """
        scores = {}

        # Indoor detection
        indoor_score = 0.0
        # Warm colors
        if visual_features["warm_color_ratio"] > 0.3:
            indoor_score += 0.3
        # Lower edge density (smooth walls)
        if visual_features["edge_density"] < 0.15:
            indoor_score += 0.2
        # Moderate brightness
        if 100 < visual_features["mean_brightness"] < 200:
            indoor_score += 0.2

        # Outdoor detection
        outdoor_score = 0.0
        # Cool colors
        if visual_features["cool_color_ratio"] > 0.3:
            outdoor_score += 0.3
        # Bright sky
        if visual_features["sky_brightness"] > 200:
            outdoor_score += 0.3
        # Natural saturation
        if 50 < visual_features["mean_saturation"] < 150:
            outdoor_score += 0.2

        # Special scene detection
        special_score = 0.0
        # High saturation
        if visual_features["mean_saturation"] > 150:
            special_score += 0.4
        # High brightness variance
        if visual_features["brightness_std"] > 60:
            special_score += 0.3

        # Determine location type
        if special_score > 0.5:
            if visual_features["mean_saturation"] > 180:
                scores["special_summon"] = 0.8
            else:
                scores["special_battle"] = 0.7
        elif indoor_score > outdoor_score:
            if visual_features["edge_density"] < 0.1:
                scores["indoor_home"] = 0.7
            elif visual_features["texture_complexity"] > 100:
                scores["indoor_school"] = 0.6
            else:
                scores["indoor_building"] = 0.6
        else:
            if visual_features["cool_color_ratio"] > 0.4:
                if visual_features["edge_density"] > 0.2:
                    scores["outdoor_street"] = 0.7
                else:
                    scores["outdoor_park"] = 0.6
            else:
                scores["outdoor_forest"] = 0.6

        return scores

    def classify_time(self, visual_features: Dict) -> Dict[str, float]:
        """
        Classify time of day

        Returns:
            Dict of {time_type: confidence}
        """
        scores = {}

        brightness = visual_features["mean_brightness"]
        warm_ratio = visual_features["warm_color_ratio"]

        if brightness > 180:
            scores["day"] = 0.9
        elif brightness < 80:
            scores["night"] = 0.8
        elif 120 < brightness < 180 and warm_ratio > 0.35:
            scores["sunset"] = 0.7
        else:
            scores["indoor_lit"] = 0.6

        return scores

    def classify_activity(
        self, visual_features: Dict, audio_features: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Classify activity type

        Returns:
            Dict of {activity_type: confidence}
        """
        scores = {}

        # Visual indicators
        high_saturation = visual_features["mean_saturation"] > 120
        high_complexity = visual_features["texture_complexity"] > 150

        # Audio indicators
        high_energy = False
        fast_tempo = False
        if audio_features:
            high_energy = audio_features.get("rms_energy", 0) > 0.05
            fast_tempo = audio_features.get("tempo", 0) > 140

        # Battle scene
        battle_score = 0.0
        if high_saturation:
            battle_score += 0.3
        if high_complexity:
            battle_score += 0.2
        if high_energy:
            battle_score += 0.3
        if fast_tempo:
            battle_score += 0.2

        if battle_score > 0.5:
            scores["battle"] = battle_score
        elif high_saturation and visual_features["mean_saturation"] > 150:
            scores["event"] = 0.7
        else:
            scores["daily"] = 0.8

        return scores

    def classify_scene(
        self,
        image_path: Path,
        audio_path: Optional[Path] = None,
        frame_time: float = 0.0,
    ) -> Dict:
        """
        Classify a single scene

        Args:
            image_path: Path to scene image
            audio_path: Optional path to audio file
            frame_time: Time in video (for audio sync)

        Returns:
            Classification results
        """
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))

        # Analyze visual features
        visual_features = self.analyze_visual_features(image)

        # Analyze audio features
        audio_features = None
        if audio_path:
            audio_features = self.analyze_audio_features(
                audio_path, start_time=frame_time, duration=2.0
            )

        # Classify
        location_scores = self.classify_location(visual_features)
        time_scores = self.classify_time(visual_features)
        activity_scores = self.classify_activity(visual_features, audio_features)

        # Get top classification for each category
        location_type = (
            max(location_scores.items(), key=lambda x: x[1])[0]
            if location_scores
            else "unknown"
        )
        time_type = (
            max(time_scores.items(), key=lambda x: x[1])[0]
            if time_scores
            else "unknown"
        )
        activity_type = (
            max(activity_scores.items(), key=lambda x: x[1])[0]
            if activity_scores
            else "unknown"
        )

        return {
            "image_path": str(image_path),
            "classifications": {
                "location": {
                    "type": location_type,
                    "confidence": location_scores.get(location_type, 0.0),
                    "all_scores": location_scores,
                },
                "time": {
                    "type": time_type,
                    "confidence": time_scores.get(time_type, 0.0),
                    "all_scores": time_scores,
                },
                "activity": {
                    "type": activity_type,
                    "confidence": activity_scores.get(activity_type, 0.0),
                    "all_scores": activity_scores,
                },
            },
            "visual_features": visual_features,
            "audio_features": audio_features,
        }

    def classify_directory(
        self, input_dir: Path, output_file: Path, audio_dir: Optional[Path] = None
    ) -> Dict:
        """
        Classify all images in a directory

        Args:
            input_dir: Input directory with images
            output_file: Output JSON file
            audio_dir: Optional audio directory

        Returns:
            Classification statistics
        """
        print(f"\n🎬 Scene Type Classification")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_file}")
        print()

        # Find images
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(input_dir.glob(ext))
            image_files.extend(input_dir.rglob(ext))

        if not image_files:
            return {"success": False, "error": "No images found"}

        # Remove duplicates
        image_files = sorted(list(set(image_files)))

        print(f"Found {len(image_files)} images")

        # Classify each image
        results = []
        stats = {"location": {}, "time": {}, "activity": {}}

        for img_path in tqdm(image_files, desc="  Classifying scenes"):
            try:
                # Find corresponding audio if available
                audio_path = None
                if audio_dir:
                    audio_name = img_path.stem + ".wav"
                    audio_path = audio_dir / audio_name

                # Classify
                result = self.classify_scene(img_path, audio_path)
                results.append(result)

                # Update statistics
                for category in ["location", "time", "activity"]:
                    scene_type = result["classifications"][category]["type"]
                    if scene_type not in stats[category]:
                        stats[category][scene_type] = 0
                    stats[category][scene_type] += 1

            except Exception as e:
                print(f"⚠️  Failed to classify {img_path.name}: {e}")
                continue

        # Save results
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_scenes": len(results),
            "results": results,
            "statistics": stats,
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        # Print statistics
        print(f"\n{'='*80}")
        print("CLASSIFICATION COMPLETE")
        print(f"{'='*80}")
        print(f"  Total scenes: {len(results)}")
        print()
        print("Location distribution:")
        for loc, count in sorted(
            stats["location"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {loc}: {count}")
        print()
        print("Time distribution:")
        for time, count in sorted(
            stats["time"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {time}: {count}")
        print()
        print("Activity distribution:")
        for activity, count in sorted(
            stats["activity"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {activity}: {count}")
        print(f"{'='*80}\n")

        return {"success": True, "stats": stats, "total": len(results)}


def main():
    parser = argparse.ArgumentParser(
        description="Classify scene types with visual and audio analysis"
    )

    parser.add_argument(
        "input_dir", type=Path, help="Input directory with scene images"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output classification JSON file",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Optional audio directory (improves classification)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Processing device (default: cuda)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    # Initialize classifier
    classifier = SceneTypeClassifier(device=args.device)

    # Classify scenes
    result = classifier.classify_directory(
        input_dir=args.input_dir, output_file=args.output_json, audio_dir=args.audio_dir
    )

    if not result["success"]:
        print(f"❌ {result.get('error', 'Classification failed')}")
        return

    print(f"Classification saved: {args.output_json}\n")


if __name__ == "__main__":
    main()
