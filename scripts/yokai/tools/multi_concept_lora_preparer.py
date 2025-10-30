#!/usr/bin/env python3
"""
Multi-Concept LoRA Training Data Preparer

Prepares training data for multi-character/style LoRAs:
- Groups characters by style/type/attribute
- Unified hierarchical trigger word system
- Sample balancing across characters
- Style-specific training parameters
- Support for concept LoRA and style LoRA

Enables training like "all cat-type yokai" or "cute yokai style" LoRAs.
"""

import shutil
from pathlib import Path
import argparse
from typing import List, Dict
import json
from datetime import datetime
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MultiConceptLoRAPreparer:
    """Prepares multi-concept LoRA training data"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def load_taxonomy(self, taxonomy_file: Path) -> Dict:
        """Load style taxonomy from classifier output"""
        with open(taxonomy_file, 'r') as f:
            return json.load(f)

    def group_by_criteria(
        self,
        taxonomy: Dict,
        group_by: str,
        group_values: List[str]
    ) -> List[str]:
        """
        Group clusters by classification criteria

        Args:
            taxonomy: Classification taxonomy
            group_by: Category to group by (appearance/attribute/style/body_type)
            group_values: Values to match (e.g., ['animal_cat', 'animal_dog'])

        Returns:
            List of cluster names matching criteria
        """
        matched_clusters = []

        for cluster in taxonomy.get('clusters', []):
            cluster_name = cluster['cluster_name']
            classifications = cluster.get('classifications', {})

            # Check if cluster matches any of the group values
            category_labels = classifications.get(group_by, {})

            for value in group_values:
                if value in category_labels:
                    matched_clusters.append(cluster_name)
                    break

        return matched_clusters

    def balance_samples(
        self,
        clusters: Dict[str, List[Path]],
        target_per_cluster: int = 40,
        max_per_cluster: int = 80
    ) -> Dict[str, List[Path]]:
        """
        Balance sample counts across clusters

        Args:
            clusters: Dict of {cluster_name: [image_paths]}
            target_per_cluster: Target images per cluster
            max_per_cluster: Maximum images per cluster

        Returns:
            Balanced clusters
        """
        balanced = {}

        for cluster_name, images in clusters.items():
            num_images = len(images)

            if num_images <= target_per_cluster:
                # Keep all (will be augmented later if needed)
                balanced[cluster_name] = images
            elif num_images <= max_per_cluster:
                # Keep all
                balanced[cluster_name] = images
            else:
                # Downsample to max
                random.shuffle(images)
                balanced[cluster_name] = images[:max_per_cluster]

        return balanced

    def generate_trigger_words(
        self,
        group_name: str,
        cluster_name: str,
        hierarchical: bool = True
    ) -> str:
        """
        Generate hierarchical trigger words

        Args:
            group_name: Group/style name
            cluster_name: Individual cluster name
            hierarchical: Use hierarchical structure

        Returns:
            Trigger word string
        """
        if hierarchical:
            # Hierarchical: "yokai, cat-type, jibanyan"
            # or "cute-yokai, animal-type, char001"
            group_trigger = group_name.replace('_', '-')
            cluster_trigger = cluster_name.replace('cluster_', 'char')

            return f"yokai, {group_trigger}, {cluster_trigger}"
        else:
            # Flat: "cat-type-yokai-char001"
            cluster_trigger = cluster_name.replace('cluster_', 'char')
            return f"{group_name}-{cluster_trigger}"

    def calculate_training_params(
        self,
        num_clusters: int,
        total_samples: int,
        training_type: str = "concept"
    ) -> Dict:
        """
        Calculate training parameters for multi-concept LoRA

        Args:
            num_clusters: Number of characters/clusters
            total_samples: Total training images
            training_type: "concept" (learn individuals + shared) or
                          "style" (focus on shared style)

        Returns:
            Training parameters
        """
        params = {
            "network_module": "networks.lora",
            "optimizer_type": "AdamW8bit",
            "lr_scheduler": "cosine_with_restarts",
            "lr_warmup_steps": 0,
            "mixed_precision": "fp16",
            "save_precision": "fp16",
            "seed": self.seed,
            "enable_bucket": True,
            "min_bucket_reso": 320,
            "max_bucket_reso": 960,
            "bucket_reso_steps": 64,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "xformers": True
        }

        if training_type == "concept":
            # Concept LoRA: Learn individuals + shared features
            if num_clusters <= 3:
                # Few characters - can learn each well
                params.update({
                    "network_dim": 64,
                    "network_alpha": 32,
                    "unet_lr": 1e-4,
                    "text_encoder_lr": 5e-5,
                    "max_train_epochs": 20,
                    "save_every_n_epochs": 2,
                    "comment": "Multi-concept LoRA (few characters, high detail)"
                })
            elif num_clusters <= 10:
                # Medium group
                params.update({
                    "network_dim": 48,
                    "network_alpha": 24,
                    "unet_lr": 8e-5,
                    "text_encoder_lr": 4e-5,
                    "max_train_epochs": 18,
                    "save_every_n_epochs": 2,
                    "comment": "Multi-concept LoRA (medium group)"
                })
            else:
                # Large group - focus on shared features
                params.update({
                    "network_dim": 32,
                    "network_alpha": 16,
                    "unet_lr": 6e-5,
                    "text_encoder_lr": 3e-5,
                    "max_train_epochs": 15,
                    "save_every_n_epochs": 1,
                    "comment": "Multi-concept LoRA (large group)"
                })

        else:  # style
            # Style LoRA: Focus on shared visual style
            params.update({
                "network_dim": 32,
                "network_alpha": 16,
                "unet_lr": 1.2e-4,  # Higher for style learning
                "text_encoder_lr": 6e-5,
                "max_train_epochs": 15,
                "save_every_n_epochs": 1,
                "comment": "Style LoRA (shared features)"
            })

        # Adjust based on total samples
        if total_samples < 200:
            params["max_train_epochs"] += 5
        elif total_samples > 1000:
            params["max_train_epochs"] -= 3

        return params

    def prepare_multi_concept_data(
        self,
        clusters_dir: Path,
        selected_clusters: List[str],
        output_dir: Path,
        group_name: str,
        training_type: str = "concept",
        repeat_count: int = None,
        validation_split: float = 0.1
    ) -> Dict:
        """
        Prepare multi-concept training data

        Args:
            clusters_dir: Source clusters directory
            selected_clusters: List of cluster names to include
            output_dir: Output directory
            group_name: Group name (e.g., "cat_type", "cute_style")
            training_type: "concept" or "style"
            repeat_count: Optional repeat count override
            validation_split: Validation set ratio

        Returns:
            Preparation statistics
        """
        # Collect all images from selected clusters
        cluster_images = {}
        total_images = 0

        for cluster_name in selected_clusters:
            cluster_dir = clusters_dir / cluster_name
            if not cluster_dir.exists():
                print(f"⚠️  Cluster not found: {cluster_name}")
                continue

            images = list(cluster_dir.glob("*.png"))
            if images:
                cluster_images[cluster_name] = images
                total_images += len(images)

        if not cluster_images:
            return {"success": False, "error": "No valid clusters found"}

        print(f"📊 Multi-concept preparation: {group_name}")
        print(f"  Clusters: {len(cluster_images)}")
        print(f"  Total images: {total_images}")

        # Balance samples
        balanced = self.balance_samples(cluster_images)

        # Calculate parameters
        params = self.calculate_training_params(
            num_clusters=len(balanced),
            total_samples=sum(len(imgs) for imgs in balanced.values()),
            training_type=training_type
        )

        if repeat_count is None:
            repeat_count = 12 if training_type == "concept" else 10

        # Create training directory
        train_dir = output_dir / f"{repeat_count}_{group_name}"
        train_dir.mkdir(parents=True, exist_ok=True)

        val_dir = output_dir / "validation" / group_name
        val_dir.mkdir(parents=True, exist_ok=True)

        # Process each cluster
        train_count = 0
        val_count = 0

        for cluster_name, images in tqdm(balanced.items(), desc="  Processing clusters"):
            # Generate trigger word
            trigger = self.generate_trigger_words(group_name, cluster_name)

            # Split train/val
            num_val = max(1, int(len(images) * validation_split))
            random.shuffle(images)

            val_images = images[:num_val]
            train_images = images[num_val:]

            # Copy training images
            for img_path in train_images:
                # Copy image
                output_img = train_dir / f"{cluster_name}_{img_path.name}"
                shutil.copy2(img_path, output_img)

                # Copy or generate caption
                caption_file = img_path.with_suffix('.txt')
                output_caption = output_img.with_suffix('.txt')

                if caption_file.exists():
                    # Prepend trigger word to existing caption
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        original_caption = f.read().strip()

                    with open(output_caption, 'w', encoding='utf-8') as f:
                        f.write(f"{trigger}, {original_caption}")
                else:
                    # Generate basic caption with trigger
                    with open(output_caption, 'w', encoding='utf-8') as f:
                        f.write(f"{trigger}, anime character")

                train_count += 1

            # Copy validation images
            for img_path in val_images:
                output_img = val_dir / f"{cluster_name}_{img_path.name}"
                shutil.copy2(img_path, output_img)

                caption_file = img_path.with_suffix('.txt')
                output_caption = output_img.with_suffix('.txt')

                if caption_file.exists():
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        original_caption = f.read().strip()
                    with open(output_caption, 'w', encoding='utf-8') as f:
                        f.write(f"{trigger}, {original_caption}")
                else:
                    with open(output_caption, 'w', encoding='utf-8') as f:
                        f.write(f"{trigger}, anime character")

                val_count += 1

        # Generate config
        config_path = self.generate_config_toml(
            output_dir,
            group_name,
            params,
            train_dir
        )

        return {
            "success": True,
            "group_name": group_name,
            "num_clusters": len(balanced),
            "train_images": train_count,
            "val_images": val_count,
            "repeat_count": repeat_count,
            "training_type": training_type,
            "config_path": str(config_path)
        }

    def generate_config_toml(
        self,
        output_dir: Path,
        group_name: str,
        params: Dict,
        train_data_dir: Path
    ) -> Path:
        """Generate training config"""
        config_content = f"""# Multi-Concept LoRA Configuration: {group_name}
# Generated: {datetime.now().isoformat()}
# {params.get('comment', '')}

[general]
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
output_dir = "./output/{group_name}"
output_name = "{group_name}_lora"
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

        config_dir = output_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / f"{group_name}_config.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)

        return config_path


def prepare_from_taxonomy(
    clusters_dir: Path,
    taxonomy_file: Path,
    output_dir: Path,
    groups: List[Dict],
    training_type: str = "concept"
):
    """
    Prepare multiple groups from taxonomy

    Args:
        clusters_dir: Source clusters directory
        taxonomy_file: Classification taxonomy JSON
        output_dir: Output directory
        groups: List of group definitions, e.g.:
                [{"name": "cat_type", "category": "appearance", "values": ["animal_cat"]}]
        training_type: "concept" or "style"
    """
    print(f"\n{'='*80}")
    print("MULTI-CONCEPT LORA PREPARATION")
    print(f"{'='*80}\n")

    preparer = MultiConceptLoRAPreparer()

    # Load taxonomy
    taxonomy = preparer.load_taxonomy(taxonomy_file)

    results = []

    for group_def in groups:
        group_name = group_def['name']
        category = group_def['category']
        values = group_def['values']

        print(f"\n📦 Preparing group: {group_name}")
        print(f"  Category: {category}")
        print(f"  Values: {', '.join(values)}")

        # Get matching clusters
        selected = preparer.group_by_criteria(taxonomy, category, values)

        if not selected:
            print(f"  ⚠️  No clusters match criteria")
            continue

        print(f"  Matched: {len(selected)} clusters")

        # Prepare training data
        result = preparer.prepare_multi_concept_data(
            clusters_dir=clusters_dir,
            selected_clusters=selected,
            output_dir=output_dir / group_name,
            group_name=group_name,
            training_type=training_type
        )

        if result['success']:
            results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Groups prepared: {len(results)}")
    for result in results:
        print(f"\n  {result['group_name']}:")
        print(f"    Clusters: {result['num_clusters']}")
        print(f"    Training images: {result['train_images']}")
        print(f"    Validation images: {result['val_images']}")
        print(f"    Config: {result['config_path']}")
    print(f"{'='*80}\n")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_groups': len(results),
        'training_type': training_type,
        'results': results
    }

    metadata_path = output_dir / "multi_concept_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Metadata saved: {metadata_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-concept LoRA training data"
    )

    parser.add_argument(
        "clusters_dir",
        type=Path,
        help="Directory containing character clusters"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        required=True,
        help="Style taxonomy JSON from classifier"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--groups",
        type=str,
        required=True,
        help="Groups JSON file defining what to prepare"
    )
    parser.add_argument(
        "--training-type",
        type=str,
        default="concept",
        choices=["concept", "style"],
        help="Training type (default: concept)"
    )

    args = parser.parse_args()

    if not args.clusters_dir.exists():
        print(f"❌ Clusters directory not found: {args.clusters_dir}")
        return

    if not args.taxonomy.exists():
        print(f"❌ Taxonomy file not found: {args.taxonomy}")
        return

    # Load groups definition
    with open(args.groups, 'r') as f:
        groups = json.load(f)

    prepare_from_taxonomy(
        clusters_dir=args.clusters_dir,
        taxonomy_file=args.taxonomy,
        output_dir=args.output_dir,
        groups=groups,
        training_type=args.training_type
    )


if __name__ == "__main__":
    main()
