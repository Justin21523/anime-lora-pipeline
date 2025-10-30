#!/usr/bin/env python3
"""
Yokai Style Classifier

Classifies yokai/characters by visual style and attributes:
- AI-powered classification using CLIP
- Multi-dimensional categorization (appearance, attribute, style, body type)
- Interactive confirmation and adjustment
- Multi-label support
- Taxonomy export for multi-concept training

Enables training style-specific LoRAs like "all cat-type yokai" or "cute yokai".
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Set
import json
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

try:
    from transformers import (
        CLIPProcessor,
        CLIPModel,
        BlipProcessor,
        BlipForConditionalGeneration,
    )

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  transformers not available, using fallback classification")


class YokaiStyleClassifier:
    """Classifies yokai by style using AI and user input"""

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Category templates
        self.categories = {
            # 1) 外觀 / 類型
            "appearance": {
                # 你原本的
                "animal_cat": [
                    "cat",
                    "feline",
                    "kitten",
                    "nyan",
                    "-nyan",
                    "neko",
                    "charming cat",
                    "ネコ",
                    "ネコ科",
                    "猫",
                    "貓妖",
                    "貓咪",
                    "ジバニャン",
                    "コマじろう?",  # allow fuzzy
                    "cat-like creature",
                    "charming-nyan",
                    "shogunyan",
                    "baddinyan",
                ],
                "animal_dog": [
                    "dog",
                    "canine",
                    "puppy",
                    "dog-like creature",
                    "inu",
                    "イヌ",
                    "犬",
                    "hound",
                    "doggy",
                    "doge",
                    "ケルベロス",
                    "dog warrior",
                ],
                "animal_bird": [
                    "bird",
                    "avian",
                    "flying creature",
                    "winged",
                    "とり",
                    "鳥",
                    "crow",
                    "hawk",
                    "eagle",
                    "owl",
                    "pheasant",
                    "duck",
                    "penguin",
                ],
                "animal_dragon": [
                    "dragon",
                    "serpent",
                    "dragon-like",
                    "ryu",
                    "竜",
                    "ドラゴン",
                    "orochi",
                    "venoct",
                    "dragon spirit",
                ],
                # 新增：甲蟲 / 鍬形蟲
                "animal_beetle": [
                    "beetle",
                    "stag beetle",
                    "rhinoceros beetle",
                    "kabuto",
                    "kuwagata",
                    "カブトムシ",
                    "クワガタ",
                    "昆虫妖怪",
                    "insect warrior",
                    "beetler",
                    "rhinoggin",
                    "beetle yo-kai",
                ],
                # 新增：河童
                "animal_kappa": [
                    "kappa",
                    "river imp",
                    "walkappa",
                    "appak",
                    "supyo",
                    "robokapp",
                    "川の妖怪",
                    "河童",
                    "カッパ",
                    "tiger-kappa",
                    "tigappa",
                ],
                # 新增：狛犬 / 獅子狗 → Komasan line
                "animal_komainu": [
                    "komainu",
                    "lion-dog",
                    "guardian dog",
                    "shisa",
                    "獅子狗",
                    "狛犬",
                    "komasan",
                    "komajiro",
                    "komashura",
                    "hardy hound",
                    "lion-dog statue",
                    "shrine guardian",
                ],
                # 新增：老虎
                "animal_tiger": [
                    "tiger",
                    "big cat",
                    "striped cat",
                    "トラ",
                    "老虎",
                    "tigappa",
                    "tigertail",
                ],
                # 新增：獅子（跟狛犬分開）
                "animal_lion": [
                    "lion",
                    "lion-like",
                    "しし",
                    "獅子",
                    "blazion",
                    "dandoodle",
                    "king",
                    "mane",
                    "lion warrior",
                ],
                # 新增：熊
                "animal_bear": [
                    "bear",
                    "くま",
                    "熊",
                    "panda",
                    "polar bear",
                    "bear-like",
                    "wazzat-bear style",
                ],
                # 新增：狐狸 / 狐
                "animal_fox": [
                    "fox",
                    "kitsune",
                    "きつね",
                    "狐",
                    "kyubi",
                    "tamamo",
                    "nine tails",
                    "妖狐",
                    "fox spirit",
                ],
                # 新增：狼
                "animal_wolf": [
                    "wolf",
                    "ookami",
                    "オオカミ",
                    "狼",
                    "werewolf",
                    "wolf-like",
                    "inugami style",
                ],
                # 新增：猴子
                "animal_monkey": ["monkey", "saru", "サル", "猿", "ape", "gorilla"],
                # 新增：馬
                "animal_horse": [
                    "horse",
                    "pony",
                    "uma",
                    "ウマ",
                    "馬",
                    "centaur",
                    "horse-like",
                ],
                # 新增：牛 / 牛頭 / 乳牛
                "animal_cow": [
                    "cow",
                    "bull",
                    "ox",
                    "bovine",
                    "うし",
                    "牛",
                    "minotaur",
                    "ox-like",
                    "water buffalo",
                ],
                # 新增：豬 / 野豬
                "animal_pig_boar": [
                    "pig",
                    "boar",
                    "hog",
                    "いのしし",
                    "イノシシ",
                    "豬",
                    "野豬",
                    "pork",
                    "boar-like",
                ],
                # 新增：鹿
                "animal_deer": ["deer", "stag", "しka", "シカ", "鹿", "reindeer"],
                # 新增：兔
                "animal_rabbit": [
                    "rabbit",
                    "bunny",
                    "usagi",
                    "ウサギ",
                    "兔",
                    "usapyon",
                    "usa-pyon",
                    "space rabbit",
                    "space suit rabbit",
                ],
                # 新增：鼠
                "animal_rodent": [
                    "mouse",
                    "rat",
                    "hamster",
                    "rodent",
                    "ねずみ",
                    "ネズミ",
                    "鼠",
                ],
                # 新增：爬蟲類 / 鱷魚
                "animal_reptile": [
                    "reptile",
                    "lizard",
                    "gecko",
                    "chameleon",
                    "iguana",
                    "crocodile",
                    "alligator",
                    "とかげ",
                    "トカゲ",
                    "爬蟲",
                    "蜥蜴",
                ],
                # 新增：蛇（獨立出來，因為 wiki 有 snake 類） :contentReference[oaicite:6]{index=6}
                "animal_snake": [
                    "snake",
                    "serpent",
                    "へび",
                    "ヘビ",
                    "蛇",
                    "orochi style",
                    "naga",
                ],
                # 新增：烏龜
                "animal_turtle": [
                    "turtle",
                    "tortoise",
                    "kappa-shell",
                    "かめ",
                    "カメ",
                    "龜",
                    "sea turtle",
                ],
                # 新增：蛙 / 蟾蜍
                "animal_frog": [
                    "frog",
                    "toad",
                    "かえる",
                    "カエル",
                    "蛙",
                    "蟾蜍",
                    "amphibian",
                ],
                # 新增：魚
                "animal_fish": [
                    "fish",
                    "fish-like",
                    "merman",
                    "mermaid",
                    "pufferfish",
                    "goldfish",
                    "koi",
                    "carp",
                    "さかな",
                    "魚",
                    "sea creature",
                ],
                # 新增：鯊魚
                "animal_shark": [
                    "shark",
                    "whale shark",
                    "サメ",
                    "鯊魚",
                    "killer shark",
                    "shark-like",
                ],
                # 新增：章魚 / 烏賊
                "animal_octopus": [
                    "octopus",
                    "squid",
                    "takoyaki",
                    "タコ",
                    "イカ",
                    "章魚",
                    "烏賊",
                    "octo yo-kai",
                ],
                # 新增：恐龍
                "animal_dinosaur": [
                    "dinosaur",
                    "dino",
                    "t-rex",
                    "raptor",
                    "ancient beast",
                    "恐竜",
                    "きょうりゅう",
                ],
                # 你原本就有的「其他動物」
                "animal_other": [
                    "animal",
                    "creature",
                    "beast",
                    "pet",
                    "mammal",
                    "wild",
                    "beastman",
                ],
                # 原本的物件類我幫你拆細一點，因為 wiki 上物件妖怪真的超雜 :contentReference[oaicite:7]{index=7}
                "object_food": [
                    "food",
                    "edible",
                    "snack",
                    "meal",
                    "candy",
                    "sweets",
                    "dessert",
                    "ramen",
                    "sushi",
                    "oden",
                    "curry",
                    "burger",
                    "pizza",
                    "donut",
                    "ice cream",
                    "chocobar",
                    "お菓子",
                    "食べ物",
                    "食物",
                    "零食",
                ],
                "object_tool": [
                    "tool",
                    "weapon",
                    "implement",
                    "instrument",
                    "brush",
                    "umbrella",
                    "hammer",
                    "sword",
                    "katana",
                    "槍",
                    "斧",
                    "tool-like",
                ],
                "object_toy": [
                    "toy",
                    "doll",
                    "plaything",
                    "plush",
                    "figure",
                    "ぬいぐるみ",
                    "おもちゃ",
                    "玩具",
                ],
                # 新增：交通工具
                "object_vehicle": [
                    "car",
                    "truck",
                    "bus",
                    "train",
                    "bike",
                    "motorcycle",
                    "ship",
                    "airplane",
                    "vehicle",
                    "乗り物",
                    "交通工具",
                ],
                # 新增：服裝
                "object_clothing": [
                    "clothes",
                    "kimono",
                    "armor",
                    "helmet",
                    "hat",
                    "cap",
                    "mask",
                    "robe",
                    "yukata",
                    "衣服",
                    "衣裝",
                    "よろい",
                    "武具",
                ],
                # 新增：樂器
                "object_instrument": [
                    "instrument",
                    "drum",
                    "shamisen",
                    "guitar",
                    "piano",
                    "flute",
                    "microphone",
                    "idol mic",
                    "楽器",
                    "樂器",
                ],
                # 新增：植物型
                "plant": [
                    "plant",
                    "tree",
                    "flower",
                    "leaf",
                    "nature spirit",
                    "root",
                    "plant monster",
                    "草",
                    "樹",
                    "花",
                    "木",
                    "nature yo-kai",
                ],
                "humanoid": [
                    "human",
                    "humanoid",
                    "person",
                    "child",
                    "samurai",
                    "ninja",
                    "monk",
                    "priest",
                    "yukionna",
                    "oni girl",
                    "武士",
                    "忍者",
                    "人型",
                ],
                "ghost": [
                    "ghost",
                    "spirit",
                    "phantom",
                    "ethereal",
                    "yurei",
                    "幽霊",
                    "幽靈",
                    "wisp",
                    "whisper",
                    "soul",
                ],
                "abstract": [
                    "abstract",
                    "geometric",
                    "shapeless",
                    "energy",
                    "digital",
                    "concept",
                    "emotion",
                    "影",
                    "無形",
                    "無機質",
                ],
            },
            # 2) 屬性 / 元素
            "attribute": {
                "fire": [
                    "fire",
                    "flame",
                    "burning",
                    "hot",
                    "red glow",
                    "blaze",
                    "inferno",
                    "炎",
                    "火",
                    "灼熱",
                ],
                "water": [
                    "water",
                    "aquatic",
                    "ocean",
                    "blue",
                    "swimming",
                    "rain",
                    "river",
                    "wave",
                    "水",
                    "海",
                    "潮",
                ],
                "wind": [
                    "wind",
                    "air",
                    "flying",
                    "breeze",
                    "tornado",
                    "storm",
                    "gust",
                    "風",
                    "疾風",
                ],
                "thunder": [
                    "lightning",
                    "electric",
                    "thunder",
                    "spark",
                    "bolt",
                    "雷",
                    "電気",
                ],
                "earth": [
                    "earth",
                    "ground",
                    "rock",
                    "stone",
                    "mountain",
                    "sand",
                    "soil",
                    "地",
                    "岩",
                    "大地",
                ],
                "ice": [
                    "ice",
                    "frozen",
                    "cold",
                    "snow",
                    "crystal",
                    "blizzaria",
                    "吹雪姫",
                    "冰",
                ],
                "light": [
                    "light",
                    "holy",
                    "bright",
                    "radiant",
                    "divine",
                    "天",
                    "光",
                    "神聖",
                ],
                "dark": [
                    "dark",
                    "shadow",
                    "night",
                    "mysterious",
                    "black",
                    "wicked",
                    "shady",
                    "闇",
                    "黒",
                    "邪",
                ],
                "nature": [
                    "nature",
                    "plant",
                    "leaf",
                    "tree",
                    "green",
                    "wood",
                    "forest",
                    "草",
                    "木",
                    "自然",
                ],
                # 新增：毒
                "poison": ["poison", "toxic", "venom", "gas", "plague", "どく", "毒"],
                # 新增：金屬 / 機械（對應到很多機器妖怪、USApyon裝備等） :contentReference[oaicite:8]{index=8}
                "metal": [
                    "metal",
                    "steel",
                    "gear",
                    "robot",
                    "mecha",
                    "armor",
                    "機械",
                    "ロボ",
                    "鋼",
                ],
                # 新增：音
                "sound": [
                    "sound",
                    "music",
                    "song",
                    "idol",
                    "singer",
                    "instrument",
                    "ラップ",
                    "beat",
                ],
                # 新增：鬼 / 地獄 / 魔界
                "oni_demonic": [
                    "oni",
                    "demon",
                    "hell",
                    "ogre",
                    "oni king",
                    "enchma",
                    "魔王",
                    "鬼",
                    "冥",
                ],
            },
            # 3) 風格 / 氣質
            "style": {
                "cute": [
                    "cute",
                    "adorable",
                    "kawaii",
                    "sweet",
                    "lovely",
                    "ぷにっと",
                    "萌",
                    "charming",
                ],
                "cool": [
                    "cool",
                    "stylish",
                    "sleek",
                    "awesome",
                    "slick",
                    "クール",
                    "かっこいい",
                ],
                "brave": [
                    "brave",
                    "heroic",
                    "strong",
                    "powerful",
                    "warrior",
                    "勇敢",
                    "ヒーロー",
                    "samurai",
                ],
                "scary": [
                    "scary",
                    "frightening",
                    "spooky",
                    "creepy",
                    "horror",
                    "ghostly",
                    "怖い",
                ],
                "funny": [
                    "funny",
                    "comical",
                    "silly",
                    "goofy",
                    "humorous",
                    "ギャグ",
                    "おもしろい",
                ],
                "mysterious": [
                    "mysterious",
                    "enigmatic",
                    "strange",
                    "weird",
                    "occult",
                    "mystic",
                    "不可思議",
                ],
                "elegant": [
                    "elegant",
                    "graceful",
                    "refined",
                    "beautiful",
                    "princess",
                    "お嬢",
                    "麗しい",
                ],
                # 新增：機械 / 機器人
                "robotic": [
                    "robot",
                    "mechanical",
                    "robo-",
                    "mecha",
                    "android",
                    "gear",
                    "ロボ",
                    "メカ",
                ],
                # 新增：武士
                "samurai": [
                    "samurai",
                    "bushido",
                    "katana",
                    "ronin",
                    "将軍",
                    "武士",
                    "shogun",
                ],
                # 新增：忍者
                "ninja": ["ninja", "shinobi", "stealth", "くノ一", "忍者"],
                # 新增：idol / 歌手（不少貓妖都在追偶像…） :contentReference[oaicite:9]{index=9}
                "idol": [
                    "idol",
                    "singer",
                    "dance",
                    "pop star",
                    "idol unit",
                    "アイドル",
                ],
                # 新增：美式 / Merican
                "merican": [
                    "merican",
                    "usa",
                    "cowboy",
                    "american style",
                    "stars and stripes",
                    "merican yo-kai",
                ],
                # 新增：鬼 / 魔王風
                "oni": [
                    "oni",
                    "demon",
                    "evil king",
                    "oni armor",
                    "horned",
                    "鬼",
                    "魔",
                    "羅刹",
                ],
            },
            # 4) 身體型態
            "body_type": {
                "quadruped": [
                    "four-legs",
                    "quadruped",
                    "walking on four legs",
                    "beast form",
                    "四足",
                ],
                "bipedal": [
                    "two legs",
                    "standing",
                    "walking upright",
                    "humanoid stance",
                    "二足",
                ],
                "flying": [
                    "flying",
                    "winged",
                    "airborne",
                    "hovering",
                    "soaring",
                    "飛行",
                ],
                "floating": [
                    "floating",
                    "levitating",
                    "no legs",
                    "spirit form",
                    "浮遊",
                ],
                "multi_limbed": [
                    "many arms",
                    "many legs",
                    "multiple limbs",
                    "tentacles",
                    "octopus arms",
                    "多肢",
                ],
                # 新增：蛇形
                "serpentine": [
                    "snake body",
                    "long body",
                    "no limbs",
                    "serpent",
                    "へび型",
                ],
                # 新增：群體型（像一群辦公室職員的那種） :contentReference[oaicite:10]{index=10}
                "group": [
                    "group",
                    "swarm",
                    "army",
                    "band",
                    "multiple yokai together",
                    "集団",
                ],
                # 新增：巨大
                "giant": ["giant", "colossal", "big size", "巨大", "towering"],
                # 新增：載具型
                "vehicle_form": [
                    "vehicle form",
                    "car body",
                    "train body",
                    "ship body",
                    "乗り物型",
                ],
            },
        }

        # Load CLIP if available
        if CLIP_AVAILABLE:
            print("🔧 Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.clip_model.to(device)
            self.clip_model.eval()
            print("✓ CLIP loaded")
        else:
            self.clip_model = None
            self.clip_processor = None

    def classify_with_clip(
        self, image: Image.Image, category: str, threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Classify image using CLIP

        Args:
            image: PIL Image
            category: Category name (appearance/attribute/style/body_type)
            threshold: Minimum confidence threshold

        Returns:
            Dict of {label: confidence}
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            return {}

        # Get category labels
        labels = self.categories.get(category, {})
        if not labels:
            return {}

        # Prepare text prompts
        text_prompts = []
        label_names = []

        for label, descriptions in labels.items():
            # Use first description as primary
            prompt = f"a photo of {descriptions[0]}"
            text_prompts.append(prompt)
            label_names.append(label)

        # Process image and text
        inputs = self.clip_processor(
            text=text_prompts, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        # Get similarity scores
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0].cpu().numpy()

        # Build results
        results = {}
        for label, prob in zip(label_names, probs):
            if prob >= threshold:
                results[label] = float(prob)

        return results

    def classify_cluster(
        self, cluster_dir: Path, sample_size: int = 10, threshold: float = 0.3
    ) -> Dict:
        """
        Classify a character cluster

        Args:
            cluster_dir: Cluster directory
            sample_size: Number of images to sample
            threshold: Classification threshold

        Returns:
            Classification results
        """
        # Sample images
        image_files = sorted(cluster_dir.glob("*.png"))
        if not image_files:
            return {"success": False, "error": "No images found"}

        # Sample evenly
        if len(image_files) > sample_size:
            step = len(image_files) // sample_size
            sampled = image_files[::step][:sample_size]
        else:
            sampled = image_files

        # Classify each category
        category_scores = {}

        for category in self.categories.keys():
            category_results = []

            for img_path in sampled:
                try:
                    img = Image.open(img_path).convert("RGB")
                    scores = self.classify_with_clip(img, category, threshold)
                    category_results.append(scores)
                except Exception as e:
                    print(f"⚠️  Failed to process {img_path.name}: {e}")
                    continue

            # Aggregate scores (average across samples)
            if category_results:
                aggregated = {}
                for result in category_results:
                    for label, score in result.items():
                        if label not in aggregated:
                            aggregated[label] = []
                        aggregated[label].append(score)

                # Average
                category_scores[category] = {
                    label: float(np.mean(scores))
                    for label, scores in aggregated.items()
                }
            else:
                category_scores[category] = {}

        return {
            "success": True,
            "cluster_name": cluster_dir.name,
            "num_samples": len(sampled),
            "classifications": category_scores,
        }

    def classify_all_clusters(
        self, clusters_dir: Path, threshold: float = 0.3, sample_size: int = 10
    ) -> List[Dict]:
        """
        Classify all clusters

        Args:
            clusters_dir: Directory containing clusters
            threshold: Classification threshold
            sample_size: Images to sample per cluster

        Returns:
            List of classification results
        """
        cluster_dirs = sorted(
            [
                d
                for d in clusters_dir.iterdir()
                if d.is_dir() and d.name.startswith("cluster_")
            ]
        )

        if not cluster_dirs:
            print("❌ No clusters found")
            return []

        print(f"📊 Classifying {len(cluster_dirs)} clusters...")

        results = []

        for cluster_dir in tqdm(cluster_dirs, desc="Classifying clusters"):
            result = self.classify_cluster(cluster_dir, sample_size, threshold)

            if result.get("success", False):
                results.append(result)

        return results

    def interactive_review(self, classifications: List[Dict]) -> List[Dict]:
        """
        Interactive review and adjustment of classifications

        Args:
            classifications: Auto-generated classifications

        Returns:
            Reviewed classifications
        """
        print(f"\n{'='*80}")
        print("INTERACTIVE CLASSIFICATION REVIEW")
        print(f"{'='*80}\n")

        print("For each cluster, you can:")
        print("  a - Accept AI classification")
        print("  m - Modify classification")
        print("  s - Skip this cluster")
        print("  q - Quit review (save current state)")
        print()

        reviewed = []

        for i, result in enumerate(classifications):
            cluster_name = result["cluster_name"]

            print(f"\n[{i+1}/{len(classifications)}] {cluster_name}")
            print("AI Classification:")

            for category, labels in result["classifications"].items():
                if labels:
                    print(f"  {category}:")
                    for label, score in sorted(
                        labels.items(), key=lambda x: x[1], reverse=True
                    )[:3]:
                        print(f"    - {label}: {score:.2f}")

            choice = input("\nAction (a/m/s/q): ").strip().lower()

            if choice == "q":
                print("Quit review")
                break
            elif choice == "s":
                print("Skipped")
                continue
            elif choice == "m":
                # Modify
                print("\nModify classification:")
                modified = result.copy()

                for category in self.categories.keys():
                    print(f"\n{category}:")
                    print(
                        "Available labels:", ", ".join(self.categories[category].keys())
                    )
                    labels_input = input(
                        f"Enter labels (comma-separated) or press Enter to skip: "
                    ).strip()

                    if labels_input:
                        selected_labels = [l.strip() for l in labels_input.split(",")]
                        modified["classifications"][category] = {
                            label: 1.0
                            for label in selected_labels
                            if label in self.categories[category]
                        }

                reviewed.append(modified)
                print("✓ Modified")
            else:  # accept
                reviewed.append(result)
                print("✓ Accepted")

        return reviewed


def classify_and_organize(
    clusters_dir: Path,
    output_file: Path,
    threshold: float = 0.3,
    sample_size: int = 10,
    interactive: bool = True,
):
    """
    Classify all clusters and save taxonomy

    Args:
        clusters_dir: Directory containing clusters
        output_file: Output JSON file
        threshold: Classification threshold
        sample_size: Images to sample per cluster
        interactive: Enable interactive review
    """
    print(f"\n{'='*80}")
    print("YOKAI STYLE CLASSIFICATION")
    print(f"{'='*80}\n")

    print(f"Clusters directory: {clusters_dir}")
    print(f"Output file: {output_file}")
    print(f"Threshold: {threshold}")
    print(f"Sample size: {sample_size}")
    print(f"Interactive: {interactive}")
    print()

    # Initialize classifier
    classifier = YokaiStyleClassifier()

    # Classify all clusters
    classifications = classifier.classify_all_clusters(
        clusters_dir, threshold, sample_size
    )

    if not classifications:
        print("❌ No classifications generated")
        return

    # Interactive review
    if interactive:
        reviewed = classifier.interactive_review(classifications)
    else:
        reviewed = classifications

    # Generate taxonomy
    taxonomy = {
        "timestamp": datetime.now().isoformat(),
        "total_clusters": len(reviewed),
        "clusters": reviewed,
    }

    # Add statistics
    stats = {
        "by_appearance": {},
        "by_attribute": {},
        "by_style": {},
        "by_body_type": {},
    }

    for result in reviewed:
        for category, labels in result["classifications"].items():
            stat_key = f"by_{category}"
            if stat_key in stats:
                for label in labels.keys():
                    if label not in stats[stat_key]:
                        stats[stat_key][label] = 0
                    stats[stat_key][label] += 1

    taxonomy["statistics"] = stats

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(taxonomy, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("CLASSIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Classified: {len(reviewed)} clusters")
    print(f"  Output: {output_file}")
    print()
    print("Statistics:")
    for category, counts in stats.items():
        if counts:
            print(f"  {category}:")
            for label, count in sorted(
                counts.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                print(f"    {label}: {count} clusters")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Classify yokai by style and attributes"
    )

    parser.add_argument(
        "clusters_dir", type=Path, help="Directory containing character clusters"
    )
    parser.add_argument(
        "--output-json", type=Path, required=True, help="Output taxonomy JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Classification confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of images to sample per cluster (default: 10)",
    )
    parser.add_argument(
        "--no-interactive", action="store_true", help="Disable interactive review"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Processing device (default: cuda)",
    )

    args = parser.parse_args()

    if not args.clusters_dir.exists():
        print(f"❌ Clusters directory not found: {args.clusters_dir}")
        return

    classify_and_organize(
        clusters_dir=args.clusters_dir,
        output_file=args.output_json,
        threshold=args.threshold,
        sample_size=args.sample_size,
        interactive=not args.no_interactive,
    )


if __name__ == "__main__":
    main()
