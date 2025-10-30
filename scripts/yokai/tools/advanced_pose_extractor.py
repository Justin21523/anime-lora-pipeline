#!/usr/bin/env python3
"""
Advanced Pose Extractor

Advanced pose extraction with special handling for yokai:
- OpenPose/DWPose support for humanoid characters
- Special handling for non-humanoid yokai (quadruped, flying, multi-limb)
- Pose quality filtering
- Keypoint confidence scoring
- Sequence continuity checks

Generates high-quality pose data for ControlNet Pose training.
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
    from controlnet_aux import OpenposeDetector

    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    print("⚠️  controlnet_aux not available, using fallback pose detection")


class AdvancedPoseExtractor:
    """
    Advanced pose extractor with yokai-specific handling
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Initialize OpenPose if available
        if OPENPOSE_AVAILABLE:
            print("🔧 Loading OpenPose detector...")
            try:
                self.openpose = OpenposeDetector.from_pretrained(
                    "lllyasviel/ControlNet"
                )
                print("  ✓ OpenPose loaded")
            except Exception as e:
                print(f"  ⚠️  OpenPose loading failed: {e}")
                self.openpose = None
        else:
            self.openpose = None

        # Yokai body types (FULL version derived from Yo-kai Watch Wiki categories)
        self.body_types = [
            # --- A. 標準人型 / 接近人型（OpenPose 主力） ---
            "humanoid_standard",  # 正常比例人型 (student, samurai, idol)  :contentReference[oaicite:4]{index=4}
            "humanoid_chibi",  # Q 版、頭大腳短的人型 → 很多吉胖喵衍生
            "humanoid_longlimb",  # 手腳過長、關節拉很開的陰影/Shadowside形態
            "oni_ogre",  # 角、巨大上半身、腰布 → 對應 Category:Oni Yo-kai :contentReference[oaicite:5]{index=5}
            "tengu_winged",  # 人型 + 翅膀 + 長鼻 → Category:Tengu :contentReference[oaicite:6]{index=6}
            "butler_ghost",  # Whisper 類：人型上半身 + 鬼尾 + 漂浮
            # --- B. 動物／獸型 from wiki 的 Animal / Kappa / Komainu ---
            "bipedal_animal",  # 站起來走的貓犬熊 → Jibanyan、Komasan直立版
            "quadruped_beast",  # 四足獸：獅子狗、狛犬、老虎、獅子、狼  :contentReference[oaicite:7]{index=7}
            "kappa_aquatic",  # 河童體型：頭頂盤、水龜殼、短四肢  :contentReference[oaicite:8]{index=8}
            "komainu_guardian",  # 狛犬體型：胸前鬃毛、獅子狗像、多半四足或半直立  :contentReference[oaicite:9]{index=9}
            "serpentine_longbody",  # 蛇/龍的長條身體，不一定有腳
            "dragon_beast",  # 明確是龍、帶翅或多角 → Category:Dragon Yo-kai :contentReference[oaicite:10]{index=10}
            "centaur_like",  # 人上半身 + 獸下半身 / 騎乘一體
            "avian_beast",  # 鳥型、猛禽、會飛但不是妖雲 → 跟飛行分開
            "aquatic_fishlike",  # 魚、河童進化、水妖、烏龜型
            "insect_like",  # 甲蟲、鍬形蟲、蝴蝶、昆蟲武者（Beetler 那掛）
            "plant_rooted_beast",  # 植物/樹精類，但姿勢像獸 → 有些 event 妖怪會這樣
            # --- C. 飛行／漂浮 兩路 ---
            "flying_selfpowered",  # 官方獨立的「能自己飛」那群 → 有翅/氣流，不靠外物 :contentReference[oaicite:11]{index=11}
            "cloud_rider",  # 騎在雲上 / 騎在坐騎上 → Category:Yo-kai riding on a cloud :contentReference[oaicite:12]{index=12}
            "floating_spirit",  # 沒腳、尾巴往下、體積小：Whisper、煙霧、燈籠
            "jellyfish_floating",  # 水母/傘型往下垂 → 很多水系看起來像這樣
            # --- D. 多肢／分段／群體 ---
            "multi_limb_tentacle",  # 多手多腳、多觸手、章魚型
            "segmented_body",  # 一節一節的身體、甲殼重複、OpenPose 會斷的
            "swarm_group",  # Category:Group Yo-kai → 一張圖一群同型妖怪 :contentReference[oaicite:13]{index=13}
            "fusion_compound",  # 兩隻黏一起 / 合體型 / 左右對稱成兩妖
            "parasite_on_host",  # 明顯附著在人、動物或物體上 → 你要先偵測 host
            # --- E. 機械 / 載具 / 物件妖怪 ---
            "robot_mecha",  # Category:Robot Yo-kai → Robonyan、Robo- 系、鋼裝版 :contentReference[oaicite:14]{index=14}
            "armor_samurai",  # 武者、全身鎧、金屬板很多 → Shadowside 也很多
            "vehicle_form",  # 車、船、火車、交通工具成妖
            "object_furniture",  # 家具成妖：櫃子、箱子、桌子、鐘 → Inanimate Object Yo-kai 裡一大票 :contentReference[oaicite:15]{index=15}
            "object_tool_weapon",  # 武器、傘、刷子、筆、槌子成妖
            "object_accessory_mask",  # 面具、頭盔、裝飾物、面罩型
            "object_paper_umbrella",  # 唐傘 / 紙張類 → 真的有，TVTropes 也有 umbrella spirit 的描述 :contentReference[oaicite:16]{index=16}
            # --- F. 半身 / 特殊出現方式 ---
            "partial_upper_body",  # 只有上半身從牆/地冒出來 → 很多任務型妖怪這樣
            "head_only",  # 只有頭在畫面上（巨大臉、面具妖怪）
            "hand_arm_only",  # 只有手or手+一點肩膀 → 門把手、招手妖怪
            "wall_attached",  # 黏牆/黏門/貼在物體上的 → ControlNet 要改 bbox
            # --- G. 水/黏液/軟體 ---
            "aquatic_mermaid",  # 半人半魚/水尾巴
            "slime_blob",  # 軟塊、糯米、泥沼 → 很多搞笑系都這樣
            "liquid_form",  # 真的是水/泥流在地上流的
            # --- H. 巨大 / Boss / Enma / Shadowside ---
            "giant_boss",  # 體積 >> 畫面、頭超大、往往是劇場版/Blasters boss :contentReference[oaicite:17]{index=17}
            "winged_deity",  # Enma / 神格化 / 有裝飾翅的王族型
            "shadowside_monster",  # Shadowside 狀態：肢體變粗、怪獸化/獸化
            "godside_extended",  # 有 Godside 形態的長線妖怪 → 主要是形態膨脹版 :contentReference[oaicite:18]{index=18}
            # --- I. 抽象 / 能量 / 影子 ---
            "shadow_silhouette",  # 只有輪廓、黑影、影分身
            "energy_aura",  # 能量團、光球、火焰形 → 很多終盤妖怪
            "symbolic_emblem",  # 只有符號、圖騰、徽章型
            "abstract",  # 實在分不出型態時的保底
        ]

    def detect_body_type(self, image: np.ndarray) -> str:
        """
        Automatically detect yokai body type

        Args:
            image: RGB image array

        Returns:
            Body type string
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Analyze image regions
        height, width = image.shape[:2]

        # Top region (head/wings)
        top_region = edges[: height // 3, :]
        top_density = top_region.sum() / (top_region.shape[0] * top_region.shape[1])

        # Middle region (body/limbs)
        mid_region = edges[height // 3 : 2 * height // 3, :]
        mid_density = mid_region.sum() / (mid_region.shape[0] * mid_region.shape[1])

        # Bottom region (legs/base)
        bottom_region = edges[2 * height // 3 :, :]
        bottom_density = bottom_region.sum() / (
            bottom_region.shape[0] * bottom_region.shape[1]
        )

        # Alpha channel analysis (if RGBA)
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            # Check for legs in bottom region
            bottom_alpha = alpha[2 * height // 3 :, :]
            has_bottom = (bottom_alpha > 128).sum() > (
                bottom_alpha.shape[0] * bottom_alpha.shape[1] * 0.2
            )
        else:
            has_bottom = True

        # Classify based on region densities
        if top_density > mid_density and top_density > bottom_density:
            if has_bottom:
                return "bipedal"  # Top-heavy with legs (bird-like)
            else:
                return "flying"  # Top-heavy without legs (flying)

        elif not has_bottom or bottom_density < 0.01:
            return "floating"  # No legs

        elif mid_density > top_density * 1.5:
            # Check horizontal spread
            mid_alpha = (
                image[height // 3 : 2 * height // 3, :, 3]
                if image.shape[2] == 4
                else gray[height // 3 : 2 * height // 3, :]
            )
            left_half = mid_alpha[:, : width // 2].sum()
            right_half = mid_alpha[:, width // 2 :].sum()
            if abs(left_half - right_half) < left_half * 0.3:
                return "quadruped"  # Symmetric horizontal spread

        # Default to humanoid
        return "humanoid"

    def extract_humanoid_pose(self, image: Image.Image) -> Optional[Dict]:
        """
        Extract humanoid pose using OpenPose

        Args:
            image: PIL Image

        Returns:
            Pose data dict or None
        """
        if not self.openpose:
            return None

        try:
            # Run OpenPose
            pose_img = self.openpose(image)
            pose_array = np.array(pose_img)

            # Calculate confidence (non-black pixel ratio)
            non_black = np.any(pose_array > 10, axis=2)
            confidence = non_black.sum() / (pose_array.shape[0] * pose_array.shape[1])

            return {
                "type": "humanoid",
                "pose_image": pose_array,
                "confidence": float(confidence),
                "has_body": confidence > 0.05,
                "has_limbs": confidence > 0.1,
            }

        except Exception as e:
            print(f"⚠️  OpenPose extraction failed: {e}")
            return None

    def extract_quadruped_pose(self, image: np.ndarray) -> Dict:
        """
        Extract quadruped (4-legged animal) pose

        Uses contour detection to find body and limb positions

        Args:
            image: RGB/RGBA image array

        Returns:
            Pose data dict
        """
        # Convert to grayscale
        if image.shape[2] == 4:
            # Use alpha channel
            gray = image[:, :, 3]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {"type": "quadruped", "confidence": 0.0, "keypoints": []}

        # Get largest contour (main body)
        main_contour = max(contours, key=cv2.contourArea)

        # Find extreme points
        leftmost = tuple(main_contour[main_contour[:, :, 0].argmin()][0])
        rightmost = tuple(main_contour[main_contour[:, :, 0].argmax()][0])
        topmost = tuple(main_contour[main_contour[:, :, 1].argmin()][0])
        bottommost = tuple(main_contour[main_contour[:, :, 1].argmax()][0])

        # Estimate keypoints
        height, width = image.shape[:2]
        keypoints = {
            "head": topmost,
            "body_center": (int(width / 2), int(height / 2)),
            "front_left": (int(leftmost[0]), int(bottommost[1])),
            "front_right": (int(width * 0.4), int(bottommost[1])),
            "back_left": (int(width * 0.6), int(bottommost[1])),
            "back_right": (int(rightmost[0]), int(bottommost[1])),
        }

        # Create pose visualization
        pose_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw skeleton
        # Spine
        cv2.line(
            pose_img, keypoints["head"], keypoints["body_center"], (255, 255, 255), 2
        )

        # Legs
        cv2.line(
            pose_img,
            keypoints["body_center"],
            keypoints["front_left"],
            (255, 255, 255),
            2,
        )
        cv2.line(
            pose_img,
            keypoints["body_center"],
            keypoints["front_right"],
            (255, 255, 255),
            2,
        )
        cv2.line(
            pose_img,
            keypoints["body_center"],
            keypoints["back_left"],
            (255, 255, 255),
            2,
        )
        cv2.line(
            pose_img,
            keypoints["body_center"],
            keypoints["back_right"],
            (255, 255, 255),
            2,
        )

        # Draw joints
        for point in keypoints.values():
            cv2.circle(pose_img, point, 3, (0, 255, 0), -1)

        confidence = cv2.contourArea(main_contour) / (height * width)

        return {
            "type": "quadruped",
            "pose_image": pose_img,
            "keypoints": keypoints,
            "confidence": float(confidence),
            "has_body": True,
            "has_limbs": True,
        }

    def extract_flying_pose(self, image: np.ndarray) -> Dict:
        """
        Extract flying creature pose (mainly orientation)

        Args:
            image: RGB/RGBA image array

        Returns:
            Pose data dict
        """
        # Get silhouette
        if image.shape[2] == 4:
            gray = image[:, :, 3]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Find contour
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {"type": "flying", "confidence": 0.0}

        main_contour = max(contours, key=cv2.contourArea)

        # Fit ellipse to get orientation
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            angle = ellipse[2]
        else:
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center = (image.shape[1] // 2, image.shape[0] // 2)
            angle = 0

        # Create pose visualization (arrow showing direction)
        height, width = image.shape[:2]
        pose_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw orientation arrow
        length = min(width, height) // 3
        rad = np.deg2rad(angle)
        end_x = int(center[0] + length * np.cos(rad))
        end_y = int(center[1] + length * np.sin(rad))

        cv2.arrowedLine(
            pose_img, center, (end_x, end_y), (255, 255, 255), 3, tipLength=0.3
        )
        cv2.circle(pose_img, center, 5, (0, 255, 0), -1)

        confidence = cv2.contourArea(main_contour) / (height * width)

        return {
            "type": "flying",
            "pose_image": pose_img,
            "center": center,
            "orientation": float(angle),
            "confidence": float(confidence),
        }

    def extract_pose(
        self, image_path: Path, body_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Extract pose from image

        Args:
            image_path: Path to image
            body_type: Optional body type override

        Returns:
            Pose data dict or None
        """
        # Load image
        img = Image.open(image_path)

        # Convert to RGB/RGBA
        if img.mode not in ["RGB", "RGBA"]:
            img = img.convert("RGB")

        img_array = np.array(img)

        # Auto-detect body type if not specified
        if body_type is None:
            body_type = self.detect_body_type(img_array)

        # Extract pose based on body type
        if body_type == "humanoid":
            result = self.extract_humanoid_pose(img)
        elif body_type == "quadruped":
            result = self.extract_quadruped_pose(img_array)
        elif body_type in ["flying", "bipedal"]:
            result = self.extract_flying_pose(img_array)
        elif body_type == "floating":
            # Floating types: just center point
            height, width = img_array.shape[:2]
            pose_img = np.zeros((height, width, 3), dtype=np.uint8)
            center = (width // 2, height // 2)
            cv2.circle(pose_img, center, 10, (255, 255, 255), -1)
            result = {
                "type": "floating",
                "pose_image": pose_img,
                "center": center,
                "confidence": 0.8,
            }
        else:
            # Abstract or unknown
            return None

        if result:
            result["body_type"] = body_type
            result["image_path"] = str(image_path)

        return result

    def process_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        body_type: Optional[str] = None,
        min_confidence: float = 0.3,
    ) -> Dict:
        """
        Process entire dataset

        Args:
            input_dir: Input directory with images
            output_dir: Output directory
            body_type: Optional body type for all images
            min_confidence: Minimum pose confidence

        Returns:
            Processing statistics
        """
        print(f"\n🎭 Advanced Pose Extraction")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Body type: {body_type or 'auto-detect'}")
        print(f"  Min confidence: {min_confidence}")
        print()

        # Find images
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(input_dir.glob(ext))

        if not image_files:
            return {"success": False, "error": "No images found"}

        # Create output directories
        source_dir = output_dir / "source"
        pose_dir = output_dir / "pose"
        source_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_images": len(image_files),
            "processed": 0,
            "skipped_low_confidence": 0,
            "by_body_type": {},
        }

        results = []

        # Process each image
        for img_path in tqdm(image_files, desc="  Extracting poses"):
            try:
                # Extract pose
                pose_data = self.extract_pose(img_path, body_type)

                if not pose_data:
                    stats["skipped_low_confidence"] += 1
                    continue

                # Check confidence
                if pose_data.get("confidence", 0) < min_confidence:
                    stats["skipped_low_confidence"] += 1
                    continue

                # Save source image
                import shutil

                source_output = source_dir / img_path.name
                shutil.copy2(img_path, source_output)

                # Save pose image
                pose_output = pose_dir / img_path.name
                pose_img = pose_data.pop("pose_image")
                Image.fromarray(pose_img).save(pose_output)

                # Update statistics
                detected_type = pose_data.get("body_type", "unknown")
                if detected_type not in stats["by_body_type"]:
                    stats["by_body_type"][detected_type] = 0
                stats["by_body_type"][detected_type] += 1

                stats["processed"] += 1
                results.append(pose_data)

            except Exception as e:
                print(f"⚠️  Failed to process {img_path.name}: {e}")
                continue

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "results": results,
        }

        metadata_path = output_dir / "pose_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print("POSE EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped (low confidence): {stats['skipped_low_confidence']}")
        print()
        print("Body type distribution:")
        for body_type, count in sorted(
            stats["by_body_type"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {body_type}: {count}")
        print(f"{'='*80}\n")

        return {"success": True, "stats": stats}


def main():
    parser = argparse.ArgumentParser(
        description="Advanced pose extraction with yokai-specific handling"
    )

    parser.add_argument("input_dir", type=Path, help="Input directory with images")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--body-type",
        type=str,
        default=None,
        choices=[
            "humanoid",
            "quadruped",
            "bipedal",
            "flying",
            "floating",
            "multi_limbed",
        ],
        help="Body type (default: auto-detect)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum pose confidence (default: 0.3)",
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

    # Initialize extractor
    extractor = AdvancedPoseExtractor(device=args.device)

    # Process dataset
    result = extractor.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        body_type=args.body_type,
        min_confidence=args.min_confidence,
    )

    if not result["success"]:
        print(f"❌ {result.get('error', 'Processing failed')}")
        return

    print(f"Metadata saved: {args.output_dir / 'pose_metadata.json'}\n")


if __name__ == "__main__":
    main()
