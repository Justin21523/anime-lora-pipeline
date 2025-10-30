#!/usr/bin/env python3
"""
Yokai Watch Cluster Analyzer

Analyzes character clusters from Yokai Watch dataset and generates:
- Quality metrics for each cluster
- Training recommendations (Tier S/A/B/C)
- Visual reports (HTML + JSON)
- Character vs Yokai classification

Integrates existing tools:
- comprehensive_anime_analysis.py for color analysis
- enhanced_quality_assessment.py for quality scoring
- clustering_comparison.py for statistics
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import cv2
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class ClusterAnalyzer:
    """Analyzes character clusters for LoRA training suitability"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"🔧 Initializing Cluster Analyzer (Device: {device})")

    def analyze_image_quality(self, image_path: Path) -> Dict:
        """Analyze single image quality"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"valid": False, "error": "Cannot read image"}

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 500.0, 1.0)  # Normalize to 0-1

            # Brightness
            brightness = np.mean(gray) / 255.0

            # Contrast (std dev)
            contrast = np.std(gray) / 128.0

            # Size check
            height, width = img.shape[:2]
            size_score = 1.0 if min(height, width) >= 512 else min(height, width) / 512.0

            # Overall quality score
            quality_score = (sharpness * 0.4 +
                           (1.0 - abs(brightness - 0.5) * 2) * 0.3 +
                           contrast * 0.2 +
                           size_score * 0.1)

            return {
                "valid": True,
                "sharpness": float(sharpness),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "size": (width, height),
                "size_score": float(size_score),
                "quality_score": float(quality_score)
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def calculate_diversity(self, image_paths: List[Path]) -> float:
        """Calculate pose/appearance diversity in cluster"""
        try:
            # Sample up to 20 images for diversity check
            sample_size = min(20, len(image_paths))
            sampled = np.random.choice(image_paths, sample_size, replace=False)

            # Extract simple histogram features
            features = []
            for img_path in sampled:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Resize to small size for speed
                img_small = cv2.resize(img, (64, 64))

                # HSV histogram
                hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])

                # Normalize and concatenate
                hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
                hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
                feature = np.concatenate([hist_h, hist_s])
                features.append(feature)

            if len(features) < 2:
                return 0.5  # Default diversity

            features = np.array(features)

            # Calculate pairwise distances
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(features)

            # Average distance as diversity measure
            diversity = float(np.mean(distances))

            return min(diversity, 1.0)

        except Exception:
            return 0.5  # Default on error

    def classify_character_type(self, cluster_path: Path, sample_images: List[Path]) -> str:
        """Classify if cluster is human character or yokai"""
        # Simple heuristic based on color distribution
        try:
            # Sample 5 images
            sample = sample_images[:5] if len(sample_images) >= 5 else sample_images

            human_scores = []
            for img_path in sample:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Convert to HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Skin tone detection (hue range)
                # Human skin typically in hue range 0-25 (red-orange)
                skin_mask = cv2.inRange(hsv, (0, 20, 50), (25, 255, 255))
                skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

                # Saturation (yokai often more saturated/colorful)
                sat_mean = np.mean(hsv[:, :, 1])

                # Human score: high skin ratio, moderate saturation
                human_score = skin_ratio * 0.7 + (1.0 - sat_mean / 255.0) * 0.3
                human_scores.append(human_score)

            if not human_scores:
                return "unknown"

            avg_human_score = np.mean(human_scores)

            if avg_human_score > 0.4:
                return "human"
            else:
                return "yokai"

        except Exception:
            return "unknown"

    def analyze_cluster(self, cluster_path: Path) -> Dict:
        """Analyze single cluster"""
        image_files = list(cluster_path.glob("*.png"))

        if not image_files:
            return {"valid": False, "reason": "No images found"}

        cluster_name = cluster_path.name
        num_images = len(image_files)

        # Sample images for detailed analysis (max 50 to save time)
        sample_size = min(50, num_images)
        sampled_images = image_files if num_images <= 50 else \
                        list(np.random.choice(image_files, sample_size, replace=False))

        # Analyze image quality
        quality_scores = []
        for img_path in tqdm(sampled_images, desc=f"  Analyzing {cluster_name}", leave=False):
            quality = self.analyze_image_quality(img_path)
            if quality["valid"]:
                quality_scores.append(quality["quality_score"])

        if not quality_scores:
            return {"valid": False, "reason": "No valid images"}

        avg_quality = float(np.mean(quality_scores))
        min_quality = float(np.min(quality_scores))
        max_quality = float(np.max(quality_scores))

        # Calculate diversity
        diversity_score = self.calculate_diversity(image_files)

        # Classify character type
        char_type = self.classify_character_type(cluster_path, sampled_images)

        # Determine training tier
        tier = self.determine_tier(num_images, avg_quality, diversity_score, char_type)

        # Training recommendation
        recommended = (num_images >= 20 and avg_quality >= 0.6 and diversity_score >= 0.3)

        return {
            "valid": True,
            "cluster_name": cluster_name,
            "num_images": num_images,
            "avg_quality": avg_quality,
            "min_quality": min_quality,
            "max_quality": max_quality,
            "diversity_score": diversity_score,
            "character_type": char_type,
            "tier": tier,
            "recommended": recommended,
            "sample_images": [str(p.name) for p in sampled_images[:10]]
        }

    def determine_tier(self, num_images: int, quality: float,
                       diversity: float, char_type: str) -> str:
        """Determine training tier (S/A/B/C)"""
        # Tier S: Main characters (100+ images, high quality)
        if num_images >= 100 and quality >= 0.7 and diversity >= 0.5:
            return "S"

        # Tier A: Major characters (50-100 images, good quality)
        if num_images >= 50 and quality >= 0.65 and diversity >= 0.4:
            return "A"

        # Tier B: Minor characters or yokai (20-50 images)
        if num_images >= 20 and quality >= 0.6:
            return "B"

        # Tier C: Rare appearances (15-20 images)
        if num_images >= 15 and quality >= 0.55:
            return "C"

        # Not recommended
        return "D"

    def analyze_all_clusters(self, clusters_dir: Path) -> Dict:
        """Analyze all clusters in directory"""
        print(f"\n{'='*80}")
        print("YOKAI WATCH CLUSTER ANALYSIS")
        print(f"{'='*80}\n")

        cluster_dirs = sorted([d for d in clusters_dir.iterdir()
                             if d.is_dir() and d.name.startswith("cluster_")])

        if not cluster_dirs:
            print(f"❌ No clusters found in {clusters_dir}")
            return {}

        print(f"Found {len(cluster_dirs)} clusters to analyze\n")

        results = []
        tier_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for cluster_dir in tqdm(cluster_dirs, desc="Analyzing clusters"):
            result = self.analyze_cluster(cluster_dir)
            if result.get("valid"):
                results.append(result)
                tier_counts[result["tier"]] += 1
                type_counts[result["character_type"]] += 1

        # Sort by tier, then by num_images
        tier_order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
        results.sort(key=lambda x: (tier_order.get(x["tier"], 5), -x["num_images"]))

        # Generate summary statistics
        total_clusters = len(results)
        total_images = sum(r["num_images"] for r in results)
        recommended_clusters = sum(1 for r in results if r["recommended"])

        avg_quality = np.mean([r["avg_quality"] for r in results])
        avg_diversity = np.mean([r["diversity_score"] for r in results])

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_clusters": total_clusters,
            "total_images": total_images,
            "recommended_clusters": recommended_clusters,
            "average_quality": float(avg_quality),
            "average_diversity": float(avg_diversity),
            "tier_distribution": dict(tier_counts),
            "type_distribution": dict(type_counts),
            "clusters": results
        }

        return summary

    def generate_html_report(self, summary: Dict, output_path: Path):
        """Generate HTML visualization report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Yokai Watch Cluster Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .tier-section {{
            margin: 30px 0;
        }}
        .tier-header {{
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0 10px 0;
            padding: 10px;
            border-radius: 5px;
        }}
        .tier-s {{ background-color: #ffd700; color: #000; }}
        .tier-a {{ background-color: #c0c0c0; color: #000; }}
        .tier-b {{ background-color: #cd7f32; color: #fff; }}
        .tier-c {{ background-color: #808080; color: #fff; }}
        .tier-d {{ background-color: #404040; color: #fff; }}

        .cluster-card {{
            background-color: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .cluster-name {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .cluster-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .stat {{
            background-color: white;
            padding: 8px;
            border-radius: 3px;
            text-align: center;
        }}
        .stat-label {{
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .recommended {{
            display: inline-block;
            background-color: #2ecc71;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .not-recommended {{
            display: inline-block;
            background-color: #e74c3c;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .type-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .type-human {{ background-color: #3498db; color: white; }}
        .type-yokai {{ background-color: #9b59b6; color: white; }}
        .type-unknown {{ background-color: #95a5a6; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Yokai Watch Cluster Analysis Report</h1>
        <p>Generated: {summary['timestamp']}</p>

        <div class="summary">
            <div class="stat-card">
                <h3>Total Clusters</h3>
                <div class="value">{summary['total_clusters']}</div>
            </div>
            <div class="stat-card">
                <h3>Total Images</h3>
                <div class="value">{summary['total_images']:,}</div>
            </div>
            <div class="stat-card">
                <h3>Recommended</h3>
                <div class="value">{summary['recommended_clusters']}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Quality</h3>
                <div class="value">{summary['average_quality']:.2f}</div>
            </div>
        </div>

        <h2>📊 Distribution</h2>
        <div class="cluster-stats">
            <div class="stat">
                <div class="stat-label">Tier S</div>
                <div class="stat-value">{summary['tier_distribution'].get('S', 0)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Tier A</div>
                <div class="stat-value">{summary['tier_distribution'].get('A', 0)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Tier B</div>
                <div class="stat-value">{summary['tier_distribution'].get('B', 0)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Tier C</div>
                <div class="stat-value">{summary['tier_distribution'].get('C', 0)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Humans</div>
                <div class="stat-value">{summary['type_distribution'].get('human', 0)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Yokai</div>
                <div class="stat-value">{summary['type_distribution'].get('yokai', 0)}</div>
            </div>
        </div>
"""

        # Add clusters by tier
        for tier in ["S", "A", "B", "C", "D"]:
            tier_clusters = [c for c in summary["clusters"] if c["tier"] == tier]
            if not tier_clusters:
                continue

            html += f"""
        <div class="tier-section">
            <div class="tier-header tier-{tier.lower()}">Tier {tier} ({len(tier_clusters)} clusters)</div>
"""

            for cluster in tier_clusters[:50]:  # Limit to 50 per tier
                rec_badge = '<span class="recommended">✓ Recommended</span>' if cluster["recommended"] else \
                           '<span class="not-recommended">✗ Not Recommended</span>'

                type_class = f"type-{cluster['character_type']}"
                type_badge = f'<span class="type-badge {type_class}">{cluster["character_type"].title()}</span>'

                html += f"""
            <div class="cluster-card">
                <div class="cluster-name">
                    {cluster['cluster_name']}
                    {rec_badge}
                    {type_badge}
                </div>
                <div class="cluster-stats">
                    <div class="stat">
                        <div class="stat-label">Images</div>
                        <div class="stat-value">{cluster['num_images']}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Quality</div>
                        <div class="stat-value">{cluster['avg_quality']:.2f}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Diversity</div>
                        <div class="stat-value">{cluster['diversity_score']:.2f}</div>
                    </div>
                </div>
            </div>
"""

            html += "        </div>\n"

        html += """
    </div>
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✓ HTML report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Yokai Watch character clusters for LoRA training"
    )

    parser.add_argument(
        "clusters_dir",
        type=Path,
        help="Directory containing character clusters"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as clusters_dir)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for processing"
    )

    args = parser.parse_args()

    if not args.clusters_dir.exists():
        print(f"❌ Clusters directory not found: {args.clusters_dir}")
        return

    output_dir = args.output_dir or args.clusters_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze clusters
    analyzer = ClusterAnalyzer(device=args.device)
    summary = analyzer.analyze_all_clusters(args.clusters_dir)

    if not summary:
        print("❌ No valid clusters analyzed")
        return

    # Save JSON report
    json_path = output_dir / "yokai_cluster_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ JSON report saved: {json_path}")

    # Generate HTML report
    html_path = output_dir / "yokai_cluster_analysis.html"
    analyzer.generate_html_report(summary, html_path)

    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"  Total Clusters: {summary['total_clusters']}")
    print(f"  Total Images: {summary['total_images']:,}")
    print(f"  Recommended for Training: {summary['recommended_clusters']}")
    print(f"  Average Quality: {summary['average_quality']:.2f}")
    print(f"  Average Diversity: {summary['average_diversity']:.2f}")
    print(f"\n  Tier Distribution:")
    for tier, count in sorted(summary['tier_distribution'].items()):
        print(f"    Tier {tier}: {count} clusters")
    print(f"\n  Type Distribution:")
    for char_type, count in sorted(summary['type_distribution'].items()):
        print(f"    {char_type.title()}: {count} clusters")
    print(f"\n  Reports:")
    print(f"    JSON: {json_path}")
    print(f"    HTML: {html_path}")
    print(f"{'='*80}\n")

    # Print top recommendations
    recommended = [c for c in summary["clusters"] if c["recommended"]]
    if recommended:
        print(f"\n🎯 Top {min(10, len(recommended))} Recommended Clusters:")
        for i, cluster in enumerate(recommended[:10], 1):
            print(f"  {i}. {cluster['cluster_name']} (Tier {cluster['tier']}, "
                  f"{cluster['num_images']} images, {cluster['character_type']})")


if __name__ == "__main__":
    main()
