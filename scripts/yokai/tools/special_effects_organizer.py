#!/usr/bin/env python3
"""
Special Effects Organizer for Yokai Watch

Organizes and categorizes special effect scenes:
- Effect type classification (summon, attack, transformation, ambient)
- Pure effect extraction (effects without characters)
- Character + effect combinations
- Effect intensity analysis
- LoRA training data preparation for effect-specific models

Perfect for training effect-focused LoRAs that can add spectacular
Yokai Watch visual effects to generated images.

Taxonomy Version: 2025-10 Yokai schema extended version
Based on: docs/YOKAI_SCHEMA_EXTENDED.md
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
import shutil
import warnings

warnings.filterwarnings("ignore")


class SpecialEffectsOrganizer:
    """Organizes special effects for LoRA training"""

    def __init__(
        self,
        effect_intensity_threshold: float = 0.6,
        min_effect_area: float = 0.1,  # 10% of image
        max_effect_area: float = 0.9,  # 90% of image
    ):
        """
        Initialize organizer

        Args:
            effect_intensity_threshold: Minimum intensity for effect detection
            min_effect_area: Minimum effect coverage ratio
            max_effect_area: Maximum effect coverage ratio
        """
        self.effect_intensity_threshold = effect_intensity_threshold
        self.min_effect_area = min_effect_area
        self.max_effect_area = max_effect_area

        # Effect type keywords for classification
        self.effect_types = {
            # 1) 召喚 / 出場 / 進入
            "summon": [
                "summon",
                "summoned",
                "summoning",
                "call",
                "appear",
                "appearance",
                "entrance",
                "entry",
                "spawn",
                "materialize",
                "teleport in",
                "yo-kai medal",
                "medal",
                "watch summon",
                "召喚",
                "召喚陣",
                "出現",
                "出場",
                "召喚光柱",
                "summon beam",
                "summon circle",
                "gate open",
                "portal",
                "mirapo",
                "elevator light",
            ],
            "desummon": [
                "vanish",
                "disappear",
                "teleport out",
                "fade out",
                "退場",
                "消失",
                "seal back",
                "return to medal",
            ],
            # 2) 攻擊技 (大類)
            "attack": [
                "attack",
                "strike",
                "hit",
                "blast",
                "burst",
                "slash",
                "stab",
                "shoot",
                "projectile",
                "punch",
                "kick",
                "spin attack",
                "攻擊",
                "斬擊",
                "突刺",
                "爆擊",
                "打擊",
                "連打",
            ],
            "attack_melee": [
                "punch",
                "kick",
                "close combat",
                "claw",
                "bite",
                "tail swing",
                "近戰",
                "近距離攻擊",
                "體術",
                "武術",
            ],
            "attack_weapon": [
                "sword",
                "katana",
                "spear",
                "staff",
                "hammer",
                "club",
                "武器攻擊",
                "刀光",
                "刀氣",
                "劍氣",
                "weapon slash",
            ],
            "attack_projectile": [
                "bullet",
                "shoot",
                "arrow",
                "wave",
                "energy shot",
                "fireball",
                "projectile",
                "飛彈",
                "飛鏢",
                "投射",
                "遠程攻擊",
            ],
            "attack_beam": [
                "beam",
                "laser",
                "ray",
                "kame",
                "straight beam",
                "light beam",
                "光束",
                "雷射",
                "直線光",
                "貫通光",
            ],
            "attack_aoe": [
                "aoe",
                "area attack",
                "shockwave",
                "ground slam",
                "地裂",
                "地面衝擊",
                "circle burst",
                "radial burst",
                "範圍技",
                "全畫面攻擊",
            ],
            "attack_soultimate": [
                "soultimate",
                "soultimate move",
                "m skill",
                "big move",
                "必殺",
                "必殺技",
                "大招",
                "big finisher",
                "ultimate",
                "cut-in",
                "kanji cut-in",
                "pose cut-in",
            ],
            # 3) 附身 / 妖氣 / 咒
            "inspirit": [
                "inspirit",
                "possession",
                "haunt",
                "curse",
                "debuff",
                "inflict",
                "妖氣",
                "附身",
                "附體",
                "詛咒",
                "降咒",
                "狀態異常",
            ],
            "buff_support": [
                "buff",
                "support",
                "heal",
                "recovery",
                "regeneration",
                "HP up",
                "defense up",
                "atk up",
                "speed up",
                "加速",
                "治癒",
                "治療",
                "回復",
                "強化",
                "輔助",
            ],
            "status_effect": [
                "sleep",
                "paralyze",
                "confuse",
                "poison",
                "freeze",
                "burning",
                "stone",
                "slow",
                "stop",
                "狀態",
                "睡眠",
                "麻痺",
                "中毒",
                "冰凍",
                "灼傷",
            ],
            # 4) 變化 / 合成 / 進化 / Shadowside
            "transformation": [
                "transform",
                "transformation",
                "evolve",
                "evolution",
                "merge",
                "fusion",
                "合體",
                "進化",
                "變身",
                "變化",
                "變形",
                "shadowside",
                "godside",
                "awakening",
                "awaken",
                "強化形態",
            ],
            "form_change": [
                "mode change",
                "armor on",
                "power up",
                "enraged form",
                "limit break",
                "burst mode",
                "超化",
                "模式切換",
            ],
            "fusion_ritual": [
                "fusion circle",
                "fusion light",
                "fusion seal",
                "合成儀式",
                "召喚儀式",
                "儀式光陣",
            ],
            # 5) 魔法陣 / 封印 / 結界
            "magic_circle": [
                "circle",
                "magic circle",
                "seal circle",
                "summon circle",
                "rune circle",
                "glowing circle",
                "地面法陣",
                "魔法陣",
                "封印陣",
            ],
            "seal_barrier": [
                "seal",
                "barrier",
                "shield",
                "wall",
                "dome",
                "protective ring",
                "結界",
                "護盾",
                "防護罩",
                "防禦結界",
                "六角結界",
            ],
            "talisman_ofuda": [
                "ofuda",
                "paper charm",
                "talisman",
                "符",
                "符咒",
                "貼符",
                "貼紙封印",
            ],
            # 6) 元素系 (妖怪手錶很常用的顏色爆炸)
            "elemental": [
                "fire",
                "water",
                "wind",
                "lightning",
                "ice",
                "earth",
                "dark",
                "light",
                "poison",
                "shadow",
                "holy",
                "plasma",
                "flame",
                "aqua",
                "storm",
            ],
            "fire": [
                "fire",
                "flame",
                "burn",
                "fireball",
                "blaze",
                "explosion",
                "炎",
                "火焰",
                "火炎",
                "爆炎",
            ],
            "water": [
                "water",
                "bubble",
                "splash",
                "wave",
                "tide",
                "aqua jet",
                "水",
                "水流",
                "水柱",
                "水彈",
            ],
            "wind": [
                "wind",
                "gust",
                "tornado",
                "whirlwind",
                "air slash",
                "颶風",
                "旋風",
                "風刃",
            ],
            "lightning": [
                "lightning",
                "thunder",
                "electric",
                "bolt",
                "雷擊",
                "電擊",
                "雷電柱",
            ],
            "ice": [
                "ice",
                "snow",
                "frost",
                "blizzard",
                "freeze",
                "凍結",
                "冰霜",
                "冰柱",
            ],
            "earth": [
                "earth",
                "rock",
                "stone",
                "sand",
                "quake",
                "地裂",
                "落石",
                "土石",
            ],
            "darkness": [
                "dark",
                "shadow",
                "void",
                "ink",
                "corruption",
                "black flame",
                "闇",
                "黑炎",
                "影子",
                "暗黑",
            ],
            "holy_light": [
                "light",
                "holy",
                "divine",
                "radiant",
                "blessing",
                "神聖",
                "光輝",
            ],
            "poison_miasma": [
                "poison",
                "toxic",
                "miasma",
                "gas cloud",
                "purple smoke",
                "毒霧",
                "瘴氣",
                "汙染",
            ],
            # 7) 能量 / 光束 / 光環
            "energy": [
                "energy",
                "aura",
                "power",
                "charge",
                "gathering energy",
                "energy ball",
                "energy burst",
                "能量",
                "氣場",
                "光環",
                "聚氣",
            ],
            "charge_up": [
                "charge",
                "power up",
                "gather",
                "focus",
                "charging",
                "氣功",
                "蓄力",
                "集氣",
                "吸收能量",
            ],
            "aura_mode": [
                "aura",
                "flame aura",
                "dark aura",
                "angel aura",
                "妖氣外放",
                "mode aura",
                "battle aura",
            ],
            # 8) 環境 / 場景特效 (LoRA 很吃這種)
            "ambient": [
                "glow",
                "sparkle",
                "shine",
                "soft light",
                "bokeh",
                "ambient light",
                "光暈",
                "閃光",
                "發光",
                "閃爍",
                "亮粉",
            ],
            "festival_night": [
                "lantern",
                "paper lantern",
                "matsuri",
                "festival lights",
                "yokai parade",
                "夏祭",
                "燈籠",
                "夜祭",
                "煙火",
                "花火",
            ],
            "yokai_world_mist": [
                "purple mist",
                "yokai world fog",
                "mysterious haze",
                "妖界霧",
                "異界紫氣",
                "幽冥霧",
            ],
            "weather_env": [
                "rain",
                "heavy rain",
                "snow",
                "sakura petals",
                "leaf fall",
                "rain streaks",
                "rain splash",
                "下雨",
                "雪花",
                "櫻吹雪",
            ],
            "speedlines": [
                "speedline",
                "motionline",
                "battle bg",
                "rapid bg",
                "動態線",
                "戰鬥背景",
                "速度線",
            ],
            "bg_burst": [
                "radial burst",
                "impact burst",
                "comic burst",
                "背景爆光",
                "集中線",
                "爆閃",
            ],
            # 9) UI / 漫畫風 / Cut-in
            "onscreen_text": [
                "kanji cut-in",
                "text cut-in",
                "big kanji",
                "slogan",
                "on-screen text",
                "必殺文字",
                "大字",
                "效果字",
                "招式名稱",
            ],
            "comic_onomatopoeia": [
                "bam",
                "pow",
                "boom",
                "zap",
                "ドン",
                "バン",
                "ガーン",
                "擬聲字",
            ],
            "frame_overlay": [
                "vignette",
                "edge glow",
                "frame fx",
                "UI frame",
                "特效框",
                "畫面外框",
            ],
            # 10) 裝備 / 機械 / 裝置
            "device_watch": [
                "yo-kai watch",
                "watch glow",
                "watch beam",
                "dial spin",
                "手錶光",
                "錶盤旋轉",
                "召喚錶光",
            ],
            "mecha_effect": [
                "mech launch",
                "steam",
                "jet",
                "booster",
                "engine glow",
                "機械噴氣",
                "裝甲展開",
                "裝備啟動",
            ],
            # 11) 純特效圖層 (沒角色時你要標這個)
            "pure_effect": [
                "fx only",
                "no character",
                "effect plate",
                "vfx plate",
                "empty bg with glow",
                "純特效",
                "特效圖層",
            ],
        }

    def detect_pure_effect(self, image: np.ndarray) -> bool:
        """
        Detect if image contains only effects without characters

        Critical for preventing training data pollution - pure effect frames
        should be marked as 'pure_effect' and not sent to style/pose analysis.

        Args:
            image: RGB or RGBA image

        Returns:
            True if frame has only effects (no characters)
        """
        # Convert to RGB if RGBA
        if image.shape[2] == 4:
            rgb = image[:, :, :3]
            alpha = image[:, :, 3]
        else:
            rgb = image
            alpha = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

        # 1. Check for character-like features
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Detect edges - characters have structured edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Characters typically have 5-20% edge density
        # Pure effects have either very low (<3%) or very high (>25%) edge density
        has_character_edges = 0.05 < edge_density < 0.20

        # 2. Check for skin tones (common in anime characters)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Skin tone detection (hue 0-30 for human-like characters)
        skin_mask = (
            ((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 150)) &  # Red-orange hue
            (hsv[:, :, 1] > 30) &  # Some saturation
            (hsv[:, :, 1] < 180) &  # Not too saturated
            (hsv[:, :, 2] > 50) &  # Not too dark
            (hsv[:, :, 2] < 250)  # Not too bright
        )

        skin_ratio = np.sum(skin_mask) / skin_mask.size
        has_skin_tones = skin_ratio > 0.05  # 5% skin-colored pixels

        # 3. Check for typical effect patterns
        hsv_analysis = hsv.copy()

        # Effects often have extreme brightness
        very_bright = hsv_analysis[:, :, 2] > 200
        very_bright_ratio = np.sum(very_bright) / very_bright.size

        # Effects often have high saturation
        high_saturation = hsv_analysis[:, :, 1] > 200
        high_sat_ratio = np.sum(high_saturation) / high_saturation.size

        # Pure effects typically have >30% extremely bright or saturated pixels
        has_strong_effects = very_bright_ratio > 0.3 or high_sat_ratio > 0.3

        # 4. Decision logic
        # Pure effect if:
        # - No character edges AND no skin tones AND has strong effects
        # OR
        # - Has strong effects AND minimal character features
        is_pure_effect = (
            (not has_character_edges and not has_skin_tones and has_strong_effects)
            or (has_strong_effects and not has_skin_tones and edge_density < 0.03)
        )

        return is_pure_effect

    def analyze_effect_intensity(self, image: np.ndarray) -> Dict:
        """
        Analyze effect intensity in image

        Args:
            image: RGB or RGBA image

        Returns:
            Effect intensity analysis
        """
        # Convert to HSV for better effect analysis
        if image.shape[2] == 4:  # RGBA
            rgb = image[:, :, :3]
            alpha = image[:, :, 3]
        else:
            rgb = image
            alpha = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Detect bright areas (effects often have high brightness)
        brightness = hsv[:, :, 2]
        high_brightness_mask = brightness > 200

        # Detect saturated colors (colorful effects)
        saturation = hsv[:, :, 1]
        high_saturation_mask = saturation > 150

        # Detect glow/bloom effects (very bright + saturated)
        glow_mask = high_brightness_mask & high_saturation_mask

        # Calculate coverage
        total_pixels = alpha > 128  # Visible pixels
        total_count = np.sum(total_pixels)

        if total_count == 0:
            return {"has_effects": False, "intensity": 0.0, "coverage": 0.0}

        bright_count = np.sum(high_brightness_mask & total_pixels)
        saturated_count = np.sum(high_saturation_mask & total_pixels)
        glow_count = np.sum(glow_mask & total_pixels)

        bright_ratio = bright_count / total_count
        saturated_ratio = saturated_count / total_count
        glow_ratio = glow_count / total_count

        # Overall intensity score
        intensity = bright_ratio * 0.4 + saturated_ratio * 0.3 + glow_ratio * 0.3

        # Detect specific effect patterns
        has_radial = self._detect_radial_pattern(rgb)
        has_particles = self._detect_particles(rgb)
        has_waves = self._detect_wave_pattern(rgb)

        return {
            "has_effects": intensity >= self.effect_intensity_threshold,
            "intensity": float(intensity),
            "coverage": float(glow_ratio),
            "bright_ratio": float(bright_ratio),
            "saturated_ratio": float(saturated_ratio),
            "glow_ratio": float(glow_ratio),
            "has_radial": has_radial,
            "has_particles": has_particles,
            "has_waves": has_waves,
        }

    def _detect_radial_pattern(self, image: np.ndarray) -> bool:
        """Detect radial light beam patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines (radial beams often appear as lines)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10,
        )

        # Check if lines radiate from center
        if lines is not None and len(lines) > 5:
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2

            radiating_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line passes through center region
                if self._line_near_point(
                    x1, y1, x2, y2, center_x, center_y, threshold=50
                ):
                    radiating_lines += 1

            return radiating_lines >= 3

        return False

    def _line_near_point(self, x1, y1, x2, y2, px, py, threshold=50):
        """Check if line is near a point"""
        # Calculate distance from point to line
        if x2 == x1:
            dist = abs(px - x1)
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            dist = abs(m * px - py + b) / np.sqrt(m**2 + 1)

        return dist < threshold

    def _detect_particles(self, image: np.ndarray) -> bool:
        """Detect particle effects (sparkles, dust, etc.)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect small bright spots
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Count small contours (particles)
        small_particles = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 100:  # Small bright spots
                small_particles += 1

        return small_particles > 10

    def _detect_wave_pattern(self, image: np.ndarray) -> bool:
        """Detect wave/ripple patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Use Gabor filters to detect wave patterns
        # Simplified check: look for periodic brightness variations
        height, width = gray.shape

        # Check horizontal waves
        horizontal_profile = gray.mean(axis=0)
        horizontal_fft = np.fft.fft(horizontal_profile)
        horizontal_power = np.abs(horizontal_fft[1 : len(horizontal_fft) // 2])

        # Check vertical waves
        vertical_profile = gray.mean(axis=1)
        vertical_fft = np.fft.fft(vertical_profile)
        vertical_power = np.abs(vertical_fft[1 : len(vertical_fft) // 2])

        # Strong periodic pattern = wave effect
        h_max = horizontal_power.max() if len(horizontal_power) > 0 else 0
        v_max = vertical_power.max() if len(vertical_power) > 0 else 0

        return h_max > 1000 or v_max > 1000

    def separate_effect_layers(
        self, image: np.ndarray, character_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Separate effect layer from character layer

        Args:
            image: RGBA image
            character_mask: Optional character mask (if available)

        Returns:
            Dictionary with separated layers
        """
        if image.shape[2] != 4:
            # No alpha channel, can't separate
            return {"has_separation": False, "combined": image}

        rgb = image[:, :, :3]
        alpha = image[:, :, 3]

        # If character mask provided, use it
        if character_mask is not None:
            # Effect = visible pixels - character pixels
            effect_mask = (alpha > 128) & (character_mask == 0)
            character_only_mask = character_mask > 0

            # Extract layers
            effect_layer = rgb.copy()
            effect_layer[~effect_mask] = 0

            character_layer = rgb.copy()
            character_layer[~character_only_mask] = 0

            return {
                "has_separation": True,
                "effect_layer": effect_layer,
                "character_layer": character_layer,
                "effect_mask": effect_mask,
                "combined": rgb,
            }

        else:
            # Heuristic separation based on brightness/saturation
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

            # Effects tend to be very bright and saturated
            effect_mask = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] > 150) & (alpha > 128)

            # Character is everything else
            character_mask = (alpha > 128) & (~effect_mask)

            effect_layer = rgb.copy()
            effect_layer[~effect_mask] = 0

            character_layer = rgb.copy()
            character_layer[~character_mask] = 0

            return {
                "has_separation": True,
                "effect_layer": effect_layer,
                "character_layer": character_layer,
                "effect_mask": effect_mask,
                "combined": rgb,
                "note": "Heuristic separation (no character mask provided)",
            }

    def classify_effect_type(self, image: np.ndarray, metadata: Dict = None) -> Tuple[str, bool]:
        """
        Classify the type of effect

        IMPORTANT: Returns (effect_type, is_pure_effect) tuple
        When is_pure_effect=True, this frame should NOT be sent to
        style/pose analysis to prevent training data pollution.

        Args:
            image: Effect image
            metadata: Optional metadata (filename, scene info, etc.)

        Returns:
            Tuple of (effect_type, is_pure_effect)
        """
        # PRIORITY 1: Check if pure effect (no characters)
        # This is CRITICAL to prevent polluting character training data
        is_pure_effect = self.detect_pure_effect(image)

        # If pure effect, mark it immediately
        if is_pure_effect:
            # Still classify the effect type, but mark as pure
            pass  # Continue to classification below

        # Analyze visual features
        analysis = self.analyze_effect_intensity(image)

        # PRIORITY 2: Check metadata keywords (new taxonomy first)
        effect_type = None
        if metadata:
            filename = metadata.get("filename", "").lower()
            scene_info = metadata.get("scene_info", "").lower()

            text = filename + " " + scene_info

            # Check against keywords (priority order as per taxonomy)
            for effect_category, keywords in self.effect_types.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        effect_type = effect_category
                        break
                if effect_type:
                    break

        # PRIORITY 3: Visual-based classification
        if not effect_type:
            if analysis.get("has_radial", False):
                effect_type = "summon"  # Radial patterns common in summons
            elif analysis.get("has_waves", False):
                effect_type = "energy"  # Wave patterns = energy effects
            elif analysis.get("has_particles", False):
                effect_type = "ambient"  # Particles = ambient effects
            elif analysis.get("intensity", 0) > 0.8:
                effect_type = "attack"  # Very intense = attack effects
            elif analysis.get("glow_ratio", 0) > 0.3:
                effect_type = "magic_circle"  # Large glowing area = magic circle
            else:
                effect_type = "unknown"

        # If pure effect was detected, override to pure_effect
        if is_pure_effect:
            effect_type = "pure_effect"

        return effect_type, is_pure_effect

    def organize_effects(
        self, input_dir: Path, output_dir: Path, separate_layers: bool = True
    ) -> Dict:
        """
        Organize effect images by type

        Args:
            input_dir: Input directory with effect images
            output_dir: Output directory
            separate_layers: Whether to separate effect/character layers

        Returns:
            Organization statistics
        """
        print(f"\n✨ Organizing special effects")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Separate layers: {separate_layers}")
        print()

        # Find all images
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(input_dir.glob(ext))

        if not image_files:
            return {"success": False, "error": "No images found"}

        # Create output directories
        effect_types_dir = output_dir / "by_type"
        pure_effects_dir = output_dir / "pure_effects"
        combined_dir = output_dir / "combined"

        for dir_path in [effect_types_dir, pure_effects_dir, combined_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_images": len(image_files),
            "by_type": {},
            "pure_effects": 0,
            "combined": 0,
            "low_quality": 0,
            "pure_effect_types": {},  # Track pure effect subtypes
        }

        for img_path in tqdm(image_files, desc="  Processing effects"):
            try:
                # Load image
                img = Image.open(img_path)
                img_array = np.array(img)

                # Analyze
                analysis = self.analyze_effect_intensity(img_array)

                # Skip low-quality effects
                if not analysis["has_effects"]:
                    stats["low_quality"] += 1
                    continue

                # Classify (returns tuple: effect_type, is_pure_effect)
                metadata = {"filename": img_path.name}
                effect_type, is_pure_effect = self.classify_effect_type(img_array, metadata)

                # Update stats
                if effect_type not in stats["by_type"]:
                    stats["by_type"][effect_type] = 0
                stats["by_type"][effect_type] += 1

                # Track pure effects separately
                if is_pure_effect:
                    stats["pure_effects"] += 1
                    if effect_type not in stats["pure_effect_types"]:
                        stats["pure_effect_types"][effect_type] = 0
                    stats["pure_effect_types"][effect_type] += 1

                # Create type directory
                type_dir = effect_types_dir / effect_type
                type_dir.mkdir(exist_ok=True)

                # Copy to type directory
                output_path = type_dir / img_path.name
                shutil.copy2(img_path, output_path)

                # Save metadata with pure_effect flag
                meta_output = output_path.with_suffix('.json')
                with open(meta_output, 'w') as f:
                    json.dump({
                        "filename": img_path.name,
                        "effect_type": effect_type,
                        "is_pure_effect": is_pure_effect,
                        "intensity_analysis": analysis,
                        "taxonomy_version": "2025-10 Yokai schema extended",
                    }, f, indent=2)

                # Separate layers if requested
                if separate_layers and img_array.shape[2] == 4:
                    separation = self.separate_effect_layers(img_array)

                    if separation.get("has_separation", False):
                        # Save pure effect layer
                        effect_layer = separation["effect_layer"]
                        if effect_layer.max() > 0:  # Has effect content
                            effect_img = Image.fromarray(effect_layer)
                            effect_output = pure_effects_dir / effect_type
                            effect_output.mkdir(exist_ok=True)
                            effect_img.save(effect_output / img_path.name)
                            stats["pure_effects"] += 1

                        # Save combined (character + effect)
                        combined_output = combined_dir / effect_type
                        combined_output.mkdir(exist_ok=True)
                        shutil.copy2(img_path, combined_output / img_path.name)
                        stats["combined"] += 1

            except Exception as e:
                print(f"⚠️  Failed to process {img_path.name}: {e}")
                continue

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "stats": stats,
        }

        metadata_path = output_dir / "organization_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n✓ Organization complete")
        print(f"  Total processed: {stats['total_images'] - stats['low_quality']}")
        print(f"  Low quality skipped: {stats['low_quality']}")
        print(f"  Pure effects (no characters): {stats['pure_effects']}")
        print(f"  By type:")
        for effect_type, count in sorted(
            stats["by_type"].items(), key=lambda x: x[1], reverse=True
        ):
            is_pure = stats["pure_effect_types"].get(effect_type, 0)
            if is_pure > 0:
                print(f"    {effect_type}: {count} ({is_pure} pure)")
            else:
                print(f"    {effect_type}: {count}")
        print()

        return {"success": True, "stats": stats, "metadata_path": str(metadata_path)}


def organize_from_summon_scenes(
    summon_scenes_dir: Path, output_dir: Path, extract_pure_effects: bool = True
):
    """
    Organize effects from summon scene detection results

    Args:
        summon_scenes_dir: Directory with summon scene extraction
        output_dir: Output directory
        extract_pure_effects: Whether to extract pure effect layers
    """
    print(f"\n{'='*80}")
    print("SPECIAL EFFECTS ORGANIZATION")
    print(f"{'='*80}\n")

    organizer = SpecialEffectsOrganizer()

    # Find all scene directories
    scene_dirs = sorted([d for d in summon_scenes_dir.iterdir() if d.is_dir()])

    print(f"Found {len(scene_dirs)} scene directories")

    # Organize each scene
    for scene_dir in tqdm(scene_dirs, desc="Organizing scenes"):
        scene_output = output_dir / scene_dir.name

        result = organizer.organize_effects(
            input_dir=scene_dir,
            output_dir=scene_output,
            separate_layers=extract_pure_effects,
        )

    print(f"\n{'='*80}")
    print("ORGANIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Organize special effects for LoRA training"
    )

    parser.add_argument(
        "input_dir", type=Path, help="Input directory with effect images"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--separate-layers",
        action="store_true",
        help="Separate effect and character layers",
    )
    parser.add_argument(
        "--from-summon-scenes",
        action="store_true",
        help="Organize from summon scene detection output",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    if args.from_summon_scenes:
        organize_from_summon_scenes(
            summon_scenes_dir=args.input_dir,
            output_dir=args.output_dir,
            extract_pure_effects=args.separate_layers,
        )
    else:
        organizer = SpecialEffectsOrganizer()
        organizer.organize_effects(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            separate_layers=args.separate_layers,
        )


if __name__ == "__main__":
    main()
