#!/usr/bin/env python3
"""
Background LoRA Training Data Preparer for Yokai Watch

Prepares background layers for LoRA training:
- Organizes background images from layered segmentation
- Filters duplicate/similar backgrounds
- Generates scene-based captions
- Creates training directory in kohya_ss format
- Supports style-specific background training

Background LoRA training helps maintain consistent art style and scene aesthetics
when generating new character poses in existing Yokai Watch environments.
"""

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
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class BackgroundLoRAPreparer:
    """Prepares background images for LoRA training"""

    def __init__(self, similarity_threshold: float = 0.95, device: str = "cuda"):
        """
        Initialize preparer

        Args:
            similarity_threshold: Threshold for duplicate detection (0.95 = 95% similar)
            device: Processing device
        """
        self.similarity_threshold = similarity_threshold
        self.device = device

        # Load CLIP for similarity detection (if available)
        if CLIP_AVAILABLE:
            print(f"🔧 Loading CLIP model for similarity detection...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(device)
            self.clip_model.eval()
            print("✓ CLIP model loaded")
        else:
            print("⚠️  CLIP not available, using simple duplicate detection")
            self.clip_model = None
            self.clip_processor = None

    def compute_image_hash(self, img: Image.Image) -> str:
        """
        Compute perceptual hash for simple duplicate detection

        Args:
            img: PIL Image

        Returns:
            Hash string
        """
        # Resize to small size
        small = img.resize((8, 8), Image.LANCZOS).convert('L')
        pixels = np.array(small).flatten()

        # Compute average
        avg = pixels.mean()

        # Create hash
        hash_bits = (pixels > avg).astype(int)
        return ''.join(str(b) for b in hash_bits)

    def compute_clip_embedding(self, img: Image.Image) -> np.ndarray:
        """
        Compute CLIP embedding for image

        Args:
            img: PIL Image

        Returns:
            Embedding vector
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            return None

        # Process image
        inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
            embedding = embedding.cpu().numpy()[0]

        return embedding / np.linalg.norm(embedding)  # Normalize

    def filter_duplicates(self, image_files: List[Path], batch_size: int = 32) -> List[Path]:
        """
        Filter duplicate/similar backgrounds

        Args:
            image_files: List of background image paths
            batch_size: Batch size for CLIP processing

        Returns:
            Filtered list of unique backgrounds
        """
        print(f"Filtering duplicates from {len(image_files)} backgrounds...")

        if CLIP_AVAILABLE and self.clip_model is not None:
            return self._filter_duplicates_clip(image_files, batch_size)
        else:
            return self._filter_duplicates_hash(image_files)

    def _filter_duplicates_hash(self, image_files: List[Path]) -> List[Path]:
        """
        Filter duplicates using perceptual hashing

        Args:
            image_files: List of image paths

        Returns:
            Filtered list
        """
        unique_images = []
        seen_hashes = set()

        for img_path in tqdm(image_files, desc="  Hashing images"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_hash = self.compute_image_hash(img)

                if img_hash not in seen_hashes:
                    unique_images.append(img_path)
                    seen_hashes.add(img_hash)
            except Exception as e:
                print(f"⚠️  Failed to process {img_path.name}: {e}")
                continue

        print(f"✓ Filtered: {len(image_files)} → {len(unique_images)} unique backgrounds")
        return unique_images

    def _filter_duplicates_clip(self, image_files: List[Path], batch_size: int) -> List[Path]:
        """
        Filter duplicates using CLIP embeddings

        Args:
            image_files: List of image paths
            batch_size: Batch size

        Returns:
            Filtered list
        """
        # Compute embeddings
        embeddings = []
        valid_files = []

        for i in tqdm(range(0, len(image_files), batch_size), desc="  Computing embeddings"):
            batch_files = image_files[i:i + batch_size]
            batch_images = []

            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    valid_files.append(img_path)
                except Exception as e:
                    print(f"⚠️  Failed to load {img_path.name}: {e}")
                    continue

            if not batch_images:
                continue

            # Process batch
            inputs = self.clip_processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                batch_embeddings = self.clip_model.get_image_features(**inputs)
                batch_embeddings = batch_embeddings.cpu().numpy()

            # Normalize
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        # Filter based on similarity
        unique_indices = []
        used = set()

        for i in tqdm(range(len(embeddings)), desc="  Filtering similar"):
            if i in used:
                continue

            # Find similar images
            similarities = cosine_similarity([embeddings[i]], embeddings)[0]
            similar_indices = np.where(similarities > self.similarity_threshold)[0]

            # Keep first, mark others as used
            unique_indices.append(i)
            for j in similar_indices:
                if j != i:
                    used.add(j)

        unique_files = [valid_files[i] for i in unique_indices]

        print(f"✓ Filtered: {len(valid_files)} → {len(unique_files)} unique backgrounds "
              f"(threshold: {self.similarity_threshold:.2f})")

        return unique_files

    def generate_background_caption(self, img: Image.Image, scene_type: str = "anime") -> str:
        """
        Generate simple caption for background

        Args:
            img: Background image
            scene_type: Scene type (anime, outdoor, indoor, etc.)

        Returns:
            Caption string
        """
        # For backgrounds, we want generic scene descriptions
        # Actual BLIP2 captioning can be added if needed

        width, height = img.size
        aspect_ratio = width / height

        # Basic scene description
        captions = [f"{scene_type} background"]

        # Add aspect hints
        if aspect_ratio > 1.5:
            captions.append("wide scene")
        elif aspect_ratio < 0.7:
            captions.append("tall scene")

        # Analyze colors (simple)
        img_array = np.array(img.resize((100, 100)))
        avg_brightness = img_array.mean() / 255

        if avg_brightness > 0.6:
            captions.append("bright lighting")
        elif avg_brightness < 0.3:
            captions.append("dark lighting")

        return ", ".join(captions)

    def prepare_training_data(
        self,
        backgrounds_dir: Path,
        output_dir: Path,
        repeat_count: int = 10,
        validation_split: float = 0.1,
        max_backgrounds: int = None,
        scene_type: str = "anime"
    ) -> Dict:
        """
        Prepare background training data

        Args:
            backgrounds_dir: Directory containing background images
            output_dir: Output training directory
            repeat_count: Repeat count for training
            validation_split: Validation set ratio
            max_backgrounds: Maximum backgrounds to include
            scene_type: Scene type for captions

        Returns:
            Preparation statistics
        """
        print(f"\n{'='*80}")
        print("BACKGROUND LORA TRAINING DATA PREPARATION")
        print(f"{'='*80}\n")

        # Find background images
        image_files = sorted(backgrounds_dir.glob("*.jpg")) + sorted(backgrounds_dir.glob("*.png"))

        if not image_files:
            return {
                "success": False,
                "error": "No background images found"
            }

        print(f"Found {len(image_files)} background images")

        # Filter duplicates
        unique_files = self.filter_duplicates(image_files)

        # Limit if requested
        if max_backgrounds and len(unique_files) > max_backgrounds:
            print(f"Limiting to {max_backgrounds} backgrounds (from {len(unique_files)})")
            # Random sample
            import random
            random.shuffle(unique_files)
            unique_files = unique_files[:max_backgrounds]

        # Split train/validation
        num_val = max(1, int(len(unique_files) * validation_split))
        val_files = unique_files[:num_val]
        train_files = unique_files[num_val:]

        print(f"\nSplit: {len(train_files)} train, {len(val_files)} validation")

        # Create directories
        train_dir = output_dir / f"{repeat_count}_yokai_backgrounds"
        train_dir.mkdir(parents=True, exist_ok=True)

        val_dir = output_dir / "validation" / "backgrounds"
        val_dir.mkdir(parents=True, exist_ok=True)

        # Process training set
        print(f"\nPreparing training set...")
        for img_path in tqdm(train_files, desc="  Processing train"):
            try:
                # Copy image
                output_img = train_dir / img_path.name
                shutil.copy2(img_path, output_img)

                # Generate caption
                img = Image.open(img_path)
                caption = self.generate_background_caption(img, scene_type)

                # Save caption
                caption_file = output_img.with_suffix('.txt')
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)

            except Exception as e:
                print(f"⚠️  Failed to process {img_path.name}: {e}")
                continue

        # Process validation set
        print(f"Preparing validation set...")
        for img_path in tqdm(val_files, desc="  Processing validation"):
            try:
                # Copy image
                output_img = val_dir / img_path.name
                shutil.copy2(img_path, output_img)

                # Generate caption
                img = Image.open(img_path)
                caption = self.generate_background_caption(img, scene_type)

                # Save caption
                caption_file = output_img.with_suffix('.txt')
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)

            except Exception as e:
                print(f"⚠️  Failed to process {img_path.name}: {e}")
                continue

        # Generate training config
        config = self.generate_training_config(
            output_dir=output_dir,
            num_images=len(train_files),
            repeat_count=repeat_count
        )

        print(f"\n{'='*80}")
        print("PREPARATION COMPLETE")
        print(f"{'='*80}")
        print(f"  Training images: {len(train_files)}")
        print(f"  Validation images: {len(val_files)}")
        print(f"  Repeat count: {repeat_count}")
        print(f"  Output: {output_dir}")
        print(f"  Config: {config}")
        print(f"{'='*80}\n")

        return {
            "success": True,
            "total_backgrounds": len(image_files),
            "unique_backgrounds": len(unique_files),
            "train_images": len(train_files),
            "val_images": len(val_files),
            "repeat_count": repeat_count,
            "output_dir": str(output_dir),
            "config_path": str(config)
        }

    def generate_training_config(
        self,
        output_dir: Path,
        num_images: int,
        repeat_count: int
    ) -> Path:
        """
        Generate training configuration for background LoRA

        Args:
            output_dir: Output directory
            num_images: Number of training images
            repeat_count: Repeat count

        Returns:
            Config file path
        """
        # Background LoRA typically needs lower learning rates
        # and fewer epochs than character LoRA

        if num_images < 100:
            max_epochs = 15
            unet_lr = 5e-5
        elif num_images < 300:
            max_epochs = 12
            unet_lr = 8e-5
        else:
            max_epochs = 10
            unet_lr = 1e-4

        config_content = f"""# Background LoRA Training Configuration
# Generated: {datetime.now().isoformat()}

[general]
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
output_dir = "./output/yokai_backgrounds"
output_name = "yokai_backgrounds_lora"
train_data_dir = "{output_dir}"

[network]
network_module = "networks.lora"
network_dim = 32
network_alpha = 16

[training]
max_train_epochs = {max_epochs}
save_every_n_epochs = 2
mixed_precision = "fp16"
save_precision = "fp16"
seed = 42
gradient_checkpointing = true

[optimizer]
optimizer_type = "AdamW8bit"
unet_lr = {unet_lr}
text_encoder_lr = {unet_lr / 2}
lr_scheduler = "cosine"
lr_warmup_steps = 0

[resolution]
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 1024
bucket_reso_steps = 64

[optimization]
cache_latents = true
xformers = true
"""

        config_dir = output_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / "background_lora_config.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)

        return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare background images for LoRA training"
    )

    parser.add_argument(
        "backgrounds_dir",
        type=Path,
        help="Directory containing background images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for training data"
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=10,
        help="Repeat count for training (default: 10)"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--max-backgrounds",
        type=int,
        default=None,
        help="Maximum backgrounds to include (default: all unique)"
    )
    parser.add_argument(
        "--scene-type",
        type=str,
        default="anime",
        help="Scene type for captions (default: anime)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for duplicate filtering (default: 0.95)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Processing device (default: cuda)"
    )

    args = parser.parse_args()

    if not args.backgrounds_dir.exists():
        print(f"❌ Backgrounds directory not found: {args.backgrounds_dir}")
        return

    # Create preparer
    preparer = BackgroundLoRAPreparer(
        similarity_threshold=args.similarity_threshold,
        device=args.device
    )

    # Prepare training data
    result = preparer.prepare_training_data(
        backgrounds_dir=args.backgrounds_dir,
        output_dir=args.output_dir,
        repeat_count=args.repeat_count,
        validation_split=args.validation_split,
        max_backgrounds=args.max_backgrounds,
        scene_type=args.scene_type
    )

    if result["success"]:
        print(f"\n🚀 Ready to train!")
        print(f"   accelerate launch train_network.py --config_file {result['config_path']}\n")


if __name__ == "__main__":
    main()
