#!/usr/bin/env python3
"""
Clustering Method Comparison Tool
Compare different clustering algorithms and CLIP models for character identification

Supports:
- Multiple CLIP models: ViT-B/32, ViT-B/16, ViT-L/14
- Multiple clustering methods: HDBSCAN, K-means, DBSCAN, Agglomerative
- Parameter sweeps: min_cluster_size, eps, n_clusters
- Performance metrics: silhouette score, Calinski-Harabasz, Davies-Bouldin
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from hdbscan import HDBSCAN
import pandas as pd


class ClusteringComparison:
    """Compare different clustering methods and parameters"""

    def __init__(self, features: np.ndarray, labels_true: np.ndarray = None):
        """
        Initialize comparison tool

        Args:
            features: Feature vectors (N, D)
            labels_true: Optional ground truth labels for supervised metrics
        """
        self.features = features
        self.features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
        self.labels_true = labels_true
        self.results = []

    def run_hdbscan(self, min_cluster_size: int = 10, min_samples: int = 5) -> Dict:
        """Run HDBSCAN clustering"""
        print(f"  Testing HDBSCAN (min_cluster_size={min_cluster_size})...")

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(self.features_norm)

        return self._evaluate_clustering("HDBSCAN", labels, {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples
        })

    def run_kmeans(self, n_clusters: int = 10) -> Dict:
        """Run K-means clustering"""
        print(f"  Testing K-means (n_clusters={n_clusters})...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.features_norm)

        return self._evaluate_clustering("K-means", labels, {
            "n_clusters": n_clusters
        })

    def run_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> Dict:
        """Run DBSCAN clustering"""
        print(f"  Testing DBSCAN (eps={eps})...")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(self.features_norm)

        return self._evaluate_clustering("DBSCAN", labels, {
            "eps": eps,
            "min_samples": min_samples
        })

    def run_agglomerative(self, n_clusters: int = 10, linkage: str = 'ward') -> Dict:
        """Run Agglomerative clustering"""
        print(f"  Testing Agglomerative (n_clusters={n_clusters}, linkage={linkage})...")

        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agg.fit_predict(self.features_norm)

        return self._evaluate_clustering("Agglomerative", labels, {
            "n_clusters": n_clusters,
            "linkage": linkage
        })

    def _evaluate_clustering(self, method_name: str, labels: np.ndarray, params: Dict) -> Dict:
        """Evaluate clustering quality"""

        # Basic statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0

        # Cluster sizes
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))

        # Quality metrics (only for labeled points)
        labeled_mask = labels >= 0
        metrics = {}

        if labeled_mask.sum() > 1 and n_clusters > 1:
            try:
                metrics["silhouette"] = float(silhouette_score(
                    self.features_norm[labeled_mask],
                    labels[labeled_mask]
                ))
            except:
                metrics["silhouette"] = None

            try:
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(
                    self.features_norm[labeled_mask],
                    labels[labeled_mask]
                ))
            except:
                metrics["calinski_harabasz"] = None

            try:
                metrics["davies_bouldin"] = float(davies_bouldin_score(
                    self.features_norm[labeled_mask],
                    labels[labeled_mask]
                ))
            except:
                metrics["davies_bouldin"] = None

        result = {
            "method": method_name,
            "params": params,
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "noise_ratio": float(noise_ratio),
            "cluster_sizes": cluster_sizes,
            "metrics": metrics,
            "labels": labels.tolist()
        }

        self.results.append(result)
        return result

    def run_all_comparisons(self) -> List[Dict]:
        """Run comprehensive comparison"""
        print("\n🔬 Running clustering method comparisons...\n")

        # HDBSCAN with different parameters
        for min_size in [5, 10, 15, 20, 25]:
            self.run_hdbscan(min_cluster_size=min_size)

        # K-means with different cluster counts
        for n in [5, 8, 10, 12, 15]:
            self.run_kmeans(n_clusters=n)

        # DBSCAN with different eps
        for eps in [0.3, 0.4, 0.5, 0.6, 0.7]:
            self.run_dbscan(eps=eps)

        # Agglomerative clustering
        for n in [5, 8, 10, 12, 15]:
            self.run_agglomerative(n_clusters=n, linkage='ward')

        return self.results

    def generate_report(self, output_dir: Path):
        """Generate comparison report"""
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n📊 Generating comparison report...\n")

        # Save detailed results
        results_path = output_dir / "clustering_comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(self.features),
                "results": self.results
            }, f, indent=2)

        # Create summary DataFrame
        summary_data = []
        for r in self.results:
            row = {
                "Method": r["method"],
                "Params": str(r["params"]),
                "N_Clusters": r["n_clusters"],
                "Noise_Ratio": f"{r['noise_ratio']:.1%}",
                "Silhouette": r["metrics"].get("silhouette"),
                "Calinski": r["metrics"].get("calinski_harabasz"),
                "Davies": r["metrics"].get("davies_bouldin")
            }
            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Save summary CSV
        csv_path = output_dir / "clustering_comparison_summary.csv"
        df.to_csv(csv_path, index=False)

        # Print summary
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║           CLUSTERING COMPARISON SUMMARY                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        print(df.to_string(index=False))
        print()

        # Find best configurations
        self._find_best_configs()

        # Generate visualizations
        self._plot_comparison(output_dir)

        print(f"\n✓ Report saved to: {output_dir}")
        print(f"  - Detailed results: {results_path}")
        print(f"  - Summary CSV: {csv_path}")

    def _find_best_configs(self):
        """Find best configurations by different criteria"""
        print("\n🏆 Best Configurations:\n")

        # Best by silhouette score (higher is better)
        valid_results = [r for r in self.results if r["metrics"].get("silhouette") is not None]
        if valid_results:
            best_silhouette = max(valid_results, key=lambda x: x["metrics"]["silhouette"])
            print(f"Best Silhouette Score ({best_silhouette['metrics']['silhouette']:.3f}):")
            print(f"  Method: {best_silhouette['method']}")
            print(f"  Params: {best_silhouette['params']}")
            print(f"  Clusters: {best_silhouette['n_clusters']}, Noise: {best_silhouette['noise_ratio']:.1%}")
            print()

        # Best by Davies-Bouldin (lower is better)
        valid_results = [r for r in self.results if r["metrics"].get("davies_bouldin") is not None]
        if valid_results:
            best_db = min(valid_results, key=lambda x: x["metrics"]["davies_bouldin"])
            print(f"Best Davies-Bouldin Score ({best_db['metrics']['davies_bouldin']:.3f}):")
            print(f"  Method: {best_db['method']}")
            print(f"  Params: {best_db['params']}")
            print(f"  Clusters: {best_db['n_clusters']}, Noise: {best_db['noise_ratio']:.1%}")
            print()

        # Lowest noise ratio
        best_noise = min(self.results, key=lambda x: x["noise_ratio"])
        print(f"Lowest Noise Ratio ({best_noise['noise_ratio']:.1%}):")
        print(f"  Method: {best_noise['method']}")
        print(f"  Params: {best_noise['params']}")
        print(f"  Clusters: {best_noise['n_clusters']}")
        print()

    def _plot_comparison(self, output_dir: Path):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Noise ratio by method
        methods = [r["method"] for r in self.results]
        noise_ratios = [r["noise_ratio"] for r in self.results]

        ax = axes[0, 0]
        ax.scatter(range(len(methods)), noise_ratios, alpha=0.6)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Noise Ratio")
        ax.set_title("Noise Ratio Comparison")
        ax.grid(True, alpha=0.3)

        # Plot 2: Number of clusters
        n_clusters = [r["n_clusters"] for r in self.results]

        ax = axes[0, 1]
        ax.scatter(range(len(methods)), n_clusters, alpha=0.6)
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Number of Clusters")
        ax.set_title("Cluster Count Comparison")
        ax.grid(True, alpha=0.3)

        # Plot 3: Silhouette scores
        silhouette_scores = [r["metrics"].get("silhouette") for r in self.results]
        valid_indices = [i for i, s in enumerate(silhouette_scores) if s is not None]
        valid_scores = [silhouette_scores[i] for i in valid_indices]

        ax = axes[1, 0]
        if valid_scores:
            ax.scatter(valid_indices, valid_scores, alpha=0.6)
            ax.set_xlabel("Configuration")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score Comparison (higher is better)")
            ax.grid(True, alpha=0.3)

        # Plot 4: Method breakdown
        method_counts = {}
        for r in self.results:
            method = r["method"]
            method_counts[method] = method_counts.get(method, 0) + 1

        ax = axes[1, 1]
        ax.bar(method_counts.keys(), method_counts.values())
        ax.set_xlabel("Method")
        ax.set_ylabel("Number of Configurations Tested")
        ax.set_title("Methods Tested")
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plot_path = output_dir / "clustering_comparison_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  - Comparison plots: {plot_path}")


def compare_clip_models(
    image_paths: List[Path],
    models: List[str] = ["ViT-B/32", "ViT-L/14"],
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Extract features using different CLIP models

    Args:
        image_paths: List of image paths
        models: List of CLIP model names
        device: Device to use

    Returns:
        Dictionary mapping model name to features
    """
    all_features = {}

    for model_name in models:
        print(f"\n🔧 Loading CLIP model: {model_name}")
        model, preprocess = clip.load(model_name, device=device)
        model.eval()

        print(f"📊 Extracting features...")
        features_list = []

        with torch.no_grad():
            for img_path in tqdm(image_paths, desc=f"Processing with {model_name}"):
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = preprocess(image).unsqueeze(0).to(device)
                    features = model.encode_image(image_tensor)
                    features_list.append(features.cpu().numpy())
                except Exception as e:
                    print(f"\n⚠️  Failed to process {img_path.name}: {e}")
                    continue

        if features_list:
            all_features[model_name] = np.vstack(features_list)
            print(f"✓ Extracted features: {all_features[model_name].shape}")

    return all_features


def main():
    parser = argparse.ArgumentParser(
        description="Clustering Method Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare clustering methods on test dataset
  python clustering_comparison.py /tmp/clustering_test \\
    --output-dir /tmp/clustering_comparison

  # Compare CLIP models
  python clustering_comparison.py /tmp/clustering_test \\
    --output-dir /tmp/clustering_comparison \\
    --clip-models ViT-B/32 ViT-L/14

  # Quick test with limited configs
  python clustering_comparison.py /tmp/clustering_test \\
    --output-dir /tmp/clustering_comparison \\
    --quick
"""
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory with character images (episode_*/character/)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Output directory for comparison results"
    )
    parser.add_argument(
        "--clip-models",
        nargs="+",
        default=["ViT-B/32"],
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP models to compare (default: ViT-B/32)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with fewer configurations"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        help="Maximum images to process (default: 2000)"
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("CLUSTERING METHOD COMPARISON")
    print(f"{'='*80}\n")

    # Find all character images
    print("🔍 Scanning for character images...")
    character_dirs = list(args.input_dir.glob("*/character"))

    if not character_dirs:
        print(f"❌ No character directories found in {args.input_dir}")
        return

    all_images = []
    for char_dir in character_dirs:
        images = list(char_dir.glob("*.png"))
        all_images.extend(images)

    print(f"✓ Found {len(all_images)} total character images")

    # Limit images if specified
    if len(all_images) > args.max_images:
        print(f"  Limiting to {args.max_images} images for faster testing")
        np.random.seed(42)
        indices = np.random.choice(len(all_images), args.max_images, replace=False)
        all_images = [all_images[i] for i in indices]

    # Extract features with different CLIP models
    all_features = compare_clip_models(all_images, args.clip_models, args.device)

    # Run comparisons for each CLIP model
    for model_name, features in all_features.items():
        print(f"\n{'='*80}")
        print(f"COMPARING CLUSTERING METHODS WITH {model_name}")
        print(f"{'='*80}")

        # Create comparison instance
        comparison = ClusteringComparison(features)

        if args.quick:
            # Quick test mode
            print("\n🚀 Running quick comparison...\n")
            comparison.run_hdbscan(min_cluster_size=10)
            comparison.run_hdbscan(min_cluster_size=15)
            comparison.run_kmeans(n_clusters=10)
            comparison.run_dbscan(eps=0.5)
            comparison.run_agglomerative(n_clusters=10)
        else:
            # Full comparison
            comparison.run_all_comparisons()

        # Generate report
        model_output_dir = args.output_dir / model_name.replace("/", "_")
        comparison.generate_report(model_output_dir)

    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE")
    print(f"{'='*80}\n")
    print(f"Results saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
