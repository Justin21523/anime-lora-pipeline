#!/usr/bin/env python3
"""
High-Performance Parallel Character Clustering
Detects ALL characters (humans + yokai/monsters) using layered character frames

Strategy:
1. Uses layered character frames (already segmented)
2. CLIP feature extraction (works for ANY character type)
3. Parallel batch processing for maximum throughput
4. HDBSCAN clustering for automatic grouping

Perfect for Yokai Watch: detects humans, humanoid yokai, AND non-humanoid yokai
"""

import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import shutil
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')


class CharacterImageDataset(Dataset):
    """Dataset for character images"""

    def __init__(self, image_paths: List[Path], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image)
            return image_tensor, str(image_path)
        except Exception as e:
            # Return zero tensor on error
            return torch.zeros(3, 224, 224), str(image_path)


class ParallelCharacterClusterer:
    """High-performance character clustering using CLIP + HDBSCAN"""

    def __init__(
        self,
        device: str = "cuda",
        min_cluster_size: int = 25,
        batch_size: int = 64,
        num_workers: int = None
    ):
        """
        Initialize clusterer

        Args:
            device: Device for CLIP
            min_cluster_size: Minimum images per cluster
            batch_size: Batch size for feature extraction
            num_workers: DataLoader workers (default: CPU count / 2)
        """
        self.device = device
        self.min_cluster_size = min_cluster_size
        self.batch_size = batch_size

        if num_workers is None:
            self.num_workers = max(4, cpu_count() // 2)
        else:
            self.num_workers = num_workers

        print(f"🔧 Initializing High-Performance Character Clusterer")
        print(f"  Device: {device}")
        print(f"  Batch Size: {batch_size}")
        print(f"  DataLoader Workers: {self.num_workers}")
        print(f"  Min Cluster Size: {min_cluster_size}")

        # Load CLIP model
        print(f"\n📦 Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        print(f"✓ CLIP model loaded")

    def extract_features_parallel(self, image_paths: List[Path]) -> np.ndarray:
        """
        Extract CLIP features in parallel

        Args:
            image_paths: List of image paths

        Returns:
            Feature matrix (N x D)
        """
        print(f"\n🎨 Extracting features from {len(image_paths)} images...")

        # Create dataset and dataloader
        dataset = CharacterImageDataset(image_paths, self.preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
            persistent_workers=True if self.num_workers > 0 else False
        )

        features = []
        valid_paths = []

        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)

                # Extract features
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                features.append(image_features.cpu().numpy())
                valid_paths.extend(paths)

        features = np.vstack(features)

        print(f"✓ Extracted features: {features.shape}")
        return features, valid_paths

    def cluster_characters(
        self,
        features: np.ndarray,
        image_paths: List[Path]
    ) -> Dict[int, List[Path]]:
        """
        Cluster characters using HDBSCAN

        Args:
            features: Feature matrix
            image_paths: Corresponding image paths

        Returns:
            Dict mapping cluster_id -> list of image paths
        """
        print(f"\n👥 Clustering {len(features)} characters...")

        # Optional PCA for dimensionality reduction
        if features.shape[1] > 128:
            print(f"  Reducing dimensions: {features.shape[1]} -> 128")
            pca = PCA(n_components=128, random_state=42)
            features = pca.fit_transform(features)

        # HDBSCAN clustering - Optimized for detecting many distinct characters
        # Lower min_cluster_size and min_samples to detect more character groups
        clusterer = HDBSCAN(
            min_cluster_size=max(10, self.min_cluster_size // 3),  # More sensitive: ~8-10 images minimum
            min_samples=2,  # Lower threshold for core points (was 5)
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.1,  # Allow more granular clusters
            n_jobs=-1  # Use all available cores
        )

        labels = clusterer.fit_predict(features)

        # Group by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise
                continue

            if label not in clusters:
                clusters[label] = []

            clusters[label].append(Path(image_paths[idx]))

        print(f"✓ Found {len(clusters)} character clusters")
        print(f"  Noise points: {sum(1 for l in labels if l == -1)}")

        return clusters

    def save_clusters(
        self,
        clusters: Dict[int, List[Path]],
        output_dir: Path,
        copy_files: bool = False
    ):
        """
        Save clustered images to directories

        Args:
            clusters: Cluster dict
            output_dir: Output directory
            copy_files: Copy files instead of moving
        """
        print(f"\n💾 Saving clusters to: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        stats = []

        for cluster_id, paths in tqdm(sorted_clusters, desc="Saving clusters"):
            cluster_dir = output_dir / f"cluster_{cluster_id:03d}"
            cluster_dir.mkdir(exist_ok=True)

            for img_path in paths:
                dest_path = cluster_dir / img_path.name

                if copy_files:
                    shutil.copy2(img_path, dest_path)
                else:
                    shutil.move(str(img_path), str(dest_path))

            stats.append({
                "cluster_id": int(cluster_id),  # Convert numpy int to Python int
                "num_images": int(len(paths)),  # Convert to Python int
                "directory": str(cluster_dir)
            })

            print(f"  cluster_{cluster_id:03d}: {len(paths)} images")

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_clusters": len(clusters),
            "total_images": sum(len(paths) for paths in clusters.values()),
            "min_cluster_size": self.min_cluster_size,
            "clusters": stats
        }

        metadata_file = output_dir / "clustering_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Clustering complete!")
        print(f"  Clusters: {len(clusters)}")
        print(f"  Total images: {sum(len(p) for p in clusters.values())}")
        print(f"  Metadata: {metadata_file}")


def process_layered_characters(
    layered_dir: Path,
    output_dir: Path,
    min_cluster_size: int = 25,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = None,
    copy_files: bool = False
):
    """
    Process layered character frames for clustering

    Args:
        layered_dir: Directory with layered character frames
        output_dir: Output directory for clusters
        min_cluster_size: Minimum cluster size
        device: Device for processing
        batch_size: Batch size for feature extraction
        num_workers: Number of DataLoader workers
        copy_files: Copy instead of move
    """
    print(f"\n{'='*80}")
    print("HIGH-PERFORMANCE PARALLEL CHARACTER CLUSTERING")
    print(f"{'='*80}\n")

    # Find character frames
    character_dir = layered_dir / "character"
    if not character_dir.exists():
        print(f"❌ Character directory not found: {character_dir}")
        return

    image_files = sorted(character_dir.glob("*.png"))

    if not image_files:
        print(f"❌ No character images found in {character_dir}")
        return

    print(f"📊 Configuration:")
    print(f"  Input: {character_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Images: {len(image_files)}")
    print(f"  Min Cluster Size: {min_cluster_size}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Workers: {num_workers or 'auto'}")
    print(f"  Device: {device}")

    # Initialize clusterer
    clusterer = ParallelCharacterClusterer(
        device=device,
        min_cluster_size=min_cluster_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Extract features (parallel batch processing)
    start_time = datetime.now()
    features, valid_paths = clusterer.extract_features_parallel(image_files)

    # Cluster characters
    clusters = clusterer.cluster_characters(features, valid_paths)

    # Save results
    clusterer.save_clusters(clusters, output_dir, copy_files)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"✓ Total Time: {elapsed:.1f}s")
    print(f"  Feature Extraction: ~{len(image_files)/elapsed:.1f} images/sec")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="High-Performance Parallel Character Clustering"
    )

    parser.add_argument(
        "layered_dir",
        type=Path,
        help="Directory with layered segmentation output"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for clusters"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=25,
        help="Minimum images per cluster"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for processing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: CPU count / 2)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving"
    )

    args = parser.parse_args()

    process_layered_characters(
        layered_dir=args.layered_dir,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        copy_files=args.copy
    )


if __name__ == "__main__":
    main()
