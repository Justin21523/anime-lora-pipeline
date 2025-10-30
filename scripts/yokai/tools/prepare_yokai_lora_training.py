#!/usr/bin/env python3
"""
Yokai LoRA Training Data Preparer

Organizes character clusters into kohya_ss training format:
- Creates proper directory structure
- Integrates augmented images
- Generates training configuration files
- Splits validation set
- Adjusts parameters based on sample count

Supports:
- Character LoRA training
- Background LoRA training
- Small sample optimization (5-20 images)
- Multiple training frameworks (kohya_ss, etc.)
"""

import shutil
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import json
from datetime import datetime
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class LoRATrainingPreparer:
    """Prepares LoRA training data in kohya_ss format"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def create_training_structure(
        self,
        cluster_dir: Path,
        output_dir: Path,
        character_name: str,
        repeat_count: int,
        validation_split: float = 0.1,
    ) -> Dict:
        """
        Create training directory structure

        Args:
            cluster_dir: Source cluster directory
            output_dir: Output training directory
            character_name: Character name for organization
            repeat_count: Number of repeats per epoch
            validation_split: Validation set ratio

        Returns:
            Preparation statistics
        """
        # Find images and captions
        image_files = sorted(cluster_dir.glob("*.png"))

        if not image_files:
            return {"success": False, "error": "No images found"}

        # Check for captions
        caption_files = [img.with_suffix(".txt") for img in image_files]
        has_captions = all(c.exists() for c in caption_files)

        if not has_captions:
            return {
                "success": False,
                "error": "Missing captions. Run caption generation first.",
            }

        # Split train/validation
        num_val = max(1, int(len(image_files) * validation_split))
        num_val = min(num_val, len(image_files) // 10)  # Max 10% for small datasets

        indices = list(range(len(image_files)))
        random.shuffle(indices)

        val_indices = set(indices[:num_val])
        train_indices = set(indices[num_val:])

        # Create directory structure
        # Format: {repeat_count}_{character_name}
        train_dir = output_dir / f"{repeat_count}_{character_name}"
        train_dir.mkdir(parents=True, exist_ok=True)

        val_dir = output_dir / "validation" / character_name
        if num_val > 0:
            val_dir.mkdir(parents=True, exist_ok=True)

        # Copy training files
        train_count = 0
        for idx in tqdm(
            train_indices, desc=f"  Preparing {character_name}", leave=False
        ):
            img_file = image_files[idx]
            caption_file = caption_files[idx]

            # Copy image
            shutil.copy2(img_file, train_dir / img_file.name)

            # Copy caption
            shutil.copy2(caption_file, train_dir / caption_file.name)

            train_count += 1

        # Copy validation files
        val_count = 0
        for idx in val_indices:
            img_file = image_files[idx]
            caption_file = caption_files[idx]

            # Copy image
            shutil.copy2(img_file, val_dir / img_file.name)

            # Copy caption
            shutil.copy2(caption_file, val_dir / caption_file.name)

            val_count += 1

        return {
            "success": True,
            "character_name": character_name,
            "total_images": len(image_files),
            "train_images": train_count,
            "val_images": val_count,
            "repeat_count": repeat_count,
            "train_dir": str(train_dir),
            "val_dir": str(val_dir) if val_count > 0 else None,
        }

    def determine_training_params(self, num_images: int, character_type: str) -> Dict:
        """
        Determine optimal training parameters based on sample count

        Args:
            num_images: Number of training images
            character_type: "human", "yokai", or "unknown"

        Returns:
            Recommended training parameters
        """
        # Base parameters
        params = {
            "network_module": "networks.lora",
            "network_dim": 32,  # LoRA rank
            "network_alpha": 16,  # LoRA alpha
            "optimizer_type": "AdamW8bit",
            "text_encoder_lr": 5e-5,
            "unet_lr": 1e-4,
            "lr_scheduler": "cosine_with_restarts",
            "lr_warmup_steps": 0,
            "max_train_steps": None,  # Will be calculated
            "max_train_epochs": None,
            "save_every_n_epochs": 2,
            "mixed_precision": "fp16",
            "save_precision": "fp16",
            "seed": self.seed,
            "enable_bucket": True,
            "min_bucket_reso": 320,
            "max_bucket_reso": 960,
            "bucket_reso_steps": 64,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "xformers": False,
        }

        # Adjust based on sample count
        if num_images <= 10:
            # Very small dataset - need heavy training
            params.update(
                {
                    "repeat_count": 30,
                    "max_train_epochs": 30,
                    "unet_lr": 5e-5,  # Lower LR to prevent overfitting
                    "text_encoder_lr": 2e-5,
                    "save_every_n_epochs": 3,
                    "gradient_accumulation_steps": 2,
                    "comment": "Optimized for very small datasets (5-10 images)",
                }
            )
        elif num_images <= 20:
            # Small dataset
            params.update(
                {
                    "repeat_count": 20,
                    "max_train_epochs": 25,
                    "unet_lr": 8e-5,
                    "text_encoder_lr": 3e-5,
                    "save_every_n_epochs": 2,
                    "comment": "Optimized for small datasets (11-20 images)",
                }
            )
        elif num_images <= 50:
            # Medium dataset
            params.update(
                {
                    "repeat_count": 15,
                    "max_train_epochs": 20,
                    "unet_lr": 1e-4,
                    "text_encoder_lr": 5e-5,
                    "save_every_n_epochs": 2,
                    "comment": "Optimized for medium datasets (21-50 images)",
                }
            )
        elif num_images <= 100:
            # Large dataset
            params.update(
                {
                    "repeat_count": 10,
                    "max_train_epochs": 15,
                    "unet_lr": 1e-4,
                    "text_encoder_lr": 5e-5,
                    "save_every_n_epochs": 2,
                    "comment": "Optimized for large datasets (51-100 images)",
                }
            )
        else:
            # Very large dataset
            params.update(
                {
                    "repeat_count": 5,
                    "max_train_epochs": 12,
                    "unet_lr": 1e-4,
                    "text_encoder_lr": 5e-5,
                    "save_every_n_epochs": 1,
                    "comment": "Optimized for very large datasets (100+ images)",
                }
            )

        # Character type specific adjustments
        if character_type == "yokai":
            # Yokai often have more unique features, may need slightly higher LR
            params["unet_lr"] = params["unet_lr"] * 1.1
            params["network_dim"] = 48  # Higher rank for complex yokai
        elif character_type == "human":
            # Humans share more common features
            params["network_dim"] = 32

        return params

    def generate_config_toml(
        self, output_dir: Path, character_name: str, params: Dict, train_data_dir: Path
    ):
        """
        Generate kohya_ss TOML configuration file

        Args:
            output_dir: Output directory
            character_name: Character name
            params: Training parameters
            train_data_dir: Training data directory
        """
        config_content = f"""# LoRA Training Configuration for {character_name}
# Generated: {datetime.now().isoformat()}
# {params.get('comment', '')}

[general]
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
output_dir = "./output/{character_name}"
output_name = "{character_name}_lora"
train_data_dir = "{train_data_dir}"

[network]
network_module = "{params['network_module']}"
network_dim = {params['network_dim']}
network_alpha = {params['network_alpha']}

[training]
max_train_epochs = {params['max_train_epochs']}
save_every_n_epochs = {params['save_every_n_epochs']}
mixed_precision = "{params['mixed_precision']}"
save_precision = "{params['save_precision']}"
seed = {params['seed']}
gradient_checkpointing = {str(params['gradient_checkpointing']).lower()}
gradient_accumulation_steps = {params.get('gradient_accumulation_steps', 1)}

[optimizer]
optimizer_type = "{params['optimizer_type']}"
unet_lr = {params['unet_lr']}
text_encoder_lr = {params['text_encoder_lr']}
lr_scheduler = "{params['lr_scheduler']}"
lr_warmup_steps = {params['lr_warmup_steps']}

[resolution]
enable_bucket = {str(params['enable_bucket']).lower()}
min_bucket_reso = {params['min_bucket_reso']}
max_bucket_reso = {params['max_bucket_reso']}
bucket_reso_steps = {params['bucket_reso_steps']}

[optimization]
cache_latents = {str(params['cache_latents']).lower()}
xformers = {str(params['xformers']).lower()}
"""

        config_path = output_dir / f"{character_name}_config.toml"
        with open(config_path, "w") as f:
            f.write(config_content)

        return config_path


def prepare_training_data(
    clusters_dir: Path,
    output_dir: Path,
    cluster_analysis: Path = None,
    selected_clusters: List[str] = None,
    validation_split: float = 0.1,
    auto_params: bool = True,
):
    """
    Prepare training data for multiple clusters

    Args:
        clusters_dir: Directory containing clusters (with captions)
        output_dir: Output directory for training data
        cluster_analysis: Optional cluster analysis JSON
        selected_clusters: Optional list of specific clusters
        validation_split: Validation set ratio
        auto_params: Auto-determine training parameters
    """
    print(f"\n{'='*80}")
    print("YOKAI LORA TRAINING DATA PREPARATION")
    print(f"{'='*80}\n")

    # Load cluster analysis
    cluster_info = {}
    if cluster_analysis and cluster_analysis.exists():
        with open(cluster_analysis, "r") as f:
            analysis = json.load(f)
            for cluster in analysis.get("clusters", []):
                cluster_info[cluster["cluster_name"]] = {
                    "type": cluster.get("character_type", "unknown"),
                    "num_images": cluster.get("num_images", 0),
                    "recommended": cluster.get("recommended", False),
                }

    # Find clusters
    cluster_dirs = sorted(
        [
            d
            for d in clusters_dir.iterdir()
            if d.is_dir() and d.name.startswith("cluster_")
        ]
    )

    # Filter
    if selected_clusters:
        cluster_dirs = [d for d in cluster_dirs if d.name in selected_clusters]

    if not cluster_dirs:
        print("❌ No clusters found")
        return

    print(f"📊 Configuration:")
    print(f"  Clusters directory: {clusters_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Total clusters: {len(cluster_dirs)}")
    print(f"  Validation split: {validation_split:.1%}")
    print(f"  Auto parameters: {auto_params}")
    print()

    # Create preparer
    preparer = LoRATrainingPreparer()

    # Process clusters
    results = []
    for cluster_dir in tqdm(cluster_dirs, desc="Preparing training data"):
        cluster_name = cluster_dir.name

        # Get info
        info = cluster_info.get(cluster_name, {})
        char_type = info.get("type", "unknown")
        num_images = info.get("num_images", len(list(cluster_dir.glob("*.png"))))

        # Determine parameters
        if auto_params:
            params = preparer.determine_training_params(num_images, char_type)
            repeat_count = params["repeat_count"]
        else:
            repeat_count = 10  # Default
            params = preparer.determine_training_params(num_images, char_type)

        # Create character name (clean cluster name)
        character_name = cluster_name.replace("cluster_", "char_")

        # Prepare training structure
        result = preparer.create_training_structure(
            cluster_dir, output_dir, character_name, repeat_count, validation_split
        )

        if result["success"]:
            # Generate config
            train_dir = Path(result["train_dir"]).parent
            config_path = preparer.generate_config_toml(
                output_dir / "configs", character_name, params, train_dir
            )

            result["config_path"] = str(config_path)
            result["params"] = params
            result["character_type"] = char_type

            results.append(result)

    # Summary
    total_train = sum(r["train_images"] for r in results)
    total_val = sum(r["val_images"] for r in results)

    print(f"\n{'='*80}")
    print("PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Characters prepared: {len(results)}")
    print(f"  Training images: {total_train}")
    print(f"  Validation images: {total_val}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_characters": len(results),
        "total_train_images": total_train,
        "total_val_images": total_val,
        "validation_split": validation_split,
        "results": results,
    }

    metadata_path = output_dir / "preparation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Metadata saved: {metadata_path}\n")

    # Print training commands
    print("🚀 Ready to train! Example training commands:\n")
    for i, result in enumerate(results[:5], 1):
        char_name = result["character_name"]
        config = result["config_path"]
        print(f"{i}. {char_name}:")
        print(f"   accelerate launch train_network.py --config_file {config}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Yokai LoRA training data")

    parser.add_argument(
        "clusters_dir", type=Path, help="Directory containing clusters with captions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for training data",
    )
    parser.add_argument(
        "--cluster-analysis",
        type=Path,
        default=None,
        help="Path to cluster analysis JSON",
    )
    parser.add_argument(
        "--selected-clusters",
        nargs="+",
        default=None,
        help="Specific clusters to prepare",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--no-auto-params",
        action="store_true",
        help="Disable automatic parameter optimization",
    )

    args = parser.parse_args()

    if not args.clusters_dir.exists():
        print(f"❌ Clusters directory not found: {args.clusters_dir}")
        return

    prepare_training_data(
        clusters_dir=args.clusters_dir,
        output_dir=args.output_dir,
        cluster_analysis=args.cluster_analysis,
        selected_clusters=args.selected_clusters,
        validation_split=args.validation_split,
        auto_params=not args.no_auto_params,
    )


if __name__ == "__main__":
    main()
