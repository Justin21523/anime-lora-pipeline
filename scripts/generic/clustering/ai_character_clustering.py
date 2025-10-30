#!/usr/bin/env python3
"""
AI-Powered Character Clustering Tool for Anime Frames
Fully GPU-accelerated face detection and feature extraction using InsightFace

Features:
- RetinaFace for anime face detection (GPU)
- ArcFace for face feature extraction (GPU)
- HDBSCAN clustering for automatic character grouping
- Quality filtering and organization
- Visual similarity analysis

Requirements:
    pip install insightface onnxruntime-gpu scikit-learn hdbscan pillow opencv-python matplotlib seaborn
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime
import shutil
import cv2
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import insightface
from insightface.app import FaceAnalysis


class QualityFilter:
    """Filter low-quality character images"""

    def __init__(
        self,
        min_size: int = 64,
        max_blur_threshold: float = 100.0,
        min_alpha_coverage: float = 0.05,
        min_face_size: int = 32
    ):
        """
        Initialize quality filter

        Args:
            min_size: Minimum width/height in pixels
            max_blur_threshold: Maximum Laplacian variance (lower = blurrier)
            min_alpha_coverage: Minimum percentage of non-transparent pixels
            min_face_size: Minimum face size for detection
        """
        self.min_size = min_size
        self.max_blur_threshold = max_blur_threshold
        self.min_alpha_coverage = min_alpha_coverage
        self.min_face_size = min_face_size

    def is_blurry(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image is blurry using Laplacian variance

        Args:
            image: RGB or RGBA numpy array

        Returns:
            (is_blurry, blur_score)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return variance < self.max_blur_threshold, variance

    def has_enough_content(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if image has enough non-transparent pixels

        Args:
            image: RGBA numpy array

        Returns:
            (has_content, coverage_ratio)
        """
        if len(image.shape) != 3 or image.shape[2] != 4:
            return True, 1.0

        alpha = image[:, :, 3]
        coverage = (alpha > 0).sum() / alpha.size

        return coverage >= self.min_alpha_coverage, coverage

    def check_quality(self, image_path: Path, face_bbox: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive quality check

        Args:
            image_path: Path to image file
            face_bbox: Optional detected face bounding box [x1, y1, x2, y2]

        Returns:
            Dictionary with quality metrics and pass/fail status
        """
        img = np.array(Image.open(image_path))

        # Size check
        height, width = img.shape[:2]
        size_ok = width >= self.min_size and height >= self.min_size

        # Blur check
        is_blurry, blur_score = self.is_blurry(img)
        blur_ok = not is_blurry

        # Content check
        has_content, coverage = self.has_enough_content(img)

        # Face size check
        face_size_ok = True
        face_width = 0
        face_height = 0
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_width = x2 - x1
            face_height = y2 - y1
            face_size_ok = face_width >= self.min_face_size and face_height >= self.min_face_size

        return {
            "passed": bool(size_ok and blur_ok and has_content and face_size_ok),
            "width": int(width),
            "height": int(height),
            "blur_score": float(blur_score),
            "alpha_coverage": float(coverage),
            "face_width": int(face_width),
            "face_height": int(face_height),
            "reasons": {
                "size": bool(size_ok),
                "blur": bool(blur_ok),
                "content": bool(has_content),
                "face_size": bool(face_size_ok)
            }
        }


class AICharacterClusterer:
    """Main AI-powered character clustering pipeline using InsightFace"""

    def __init__(
        self,
        device: str = "cuda",
        min_cluster_size: int = 10,
        quality_filter: Optional[QualityFilter] = None,
        det_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize AI character clustering pipeline

        Args:
            device: Device for computation ('cuda' or 'cpu')
            min_cluster_size: Minimum frames per character cluster
            quality_filter: Optional quality filter instance
            det_size: Detection input size (width, height)
        """
        self.device = device
        self.min_cluster_size = min_cluster_size
        self.det_size = det_size

        print(f"\n🤖 Initializing AI Face Analysis (InsightFace)")
        print(f"   Device: {device}")
        print(f"   Detection size: {det_size}")

        # Initialize InsightFace
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']

        self.app = FaceAnalysis(
            name='buffalo_l',  # Best model for anime faces
            providers=providers
        )

        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=det_size)

        print(f"✓ InsightFace loaded (RetinaFace + ArcFace)")
        print(f"   Models: buffalo_l (optimized for faces)")

        self.quality_filter = quality_filter or QualityFilter()

    def detect_and_extract_features(
        self,
        image_paths: List[Path],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[Path], List[Dict], List[np.ndarray]]:
        """
        Detect faces and extract features using InsightFace

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing

        Returns:
            (features, valid_paths, quality_reports, face_bboxes)
        """
        features_list = []
        valid_paths = []
        quality_reports = []
        face_bboxes_list = []

        print(f"\n🔍 Detecting faces and extracting features from {len(image_paths)} images...")
        print(f"   GPU-accelerated: RetinaFace (detection) + ArcFace (features)")

        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"\n⚠️  Failed to load {img_path.name}")
                    continue

                # Convert BGR to RGB for InsightFace
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Detect faces and extract features (all GPU-accelerated)
                faces = self.app.get(img_rgb)

                if len(faces) == 0:
                    # No face detected
                    quality_reports.append({
                        "path": str(img_path),
                        "quality": {
                            "passed": False,
                            "reasons": {"face_detected": False}
                        }
                    })
                    continue

                # Use the first detected face (usually the main character)
                face = faces[0]

                # Get face bbox
                bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]

                # Quality check with face bbox
                quality = self.quality_filter.check_quality(img_path, bbox)
                quality_reports.append({
                    "path": str(img_path),
                    "quality": quality
                })

                if not quality["passed"]:
                    continue

                # Extract face embedding (512-dimensional ArcFace feature)
                embedding = face.embedding

                features_list.append(embedding)
                valid_paths.append(img_path)
                face_bboxes_list.append(bbox)

            except Exception as e:
                print(f"\n⚠️  Error processing {img_path.name}: {e}")
                continue

        if not features_list:
            raise ValueError("No valid faces detected after processing all images!")

        all_features = np.vstack(features_list)

        print(f"\n✓ Face detection and feature extraction complete:")
        print(f"   Valid faces: {len(valid_paths)} / {len(image_paths)}")
        print(f"   Feature dimensions: {all_features.shape}")
        print(f"   Filtered out: {len(image_paths) - len(valid_paths)}")

        return all_features, valid_paths, quality_reports, face_bboxes_list

    def cluster_characters(
        self,
        features: np.ndarray,
        min_cluster_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Cluster character features using HDBSCAN

        Args:
            features: Feature vectors (ArcFace embeddings)
            min_cluster_size: Minimum cluster size (overrides default)

        Returns:
            Cluster labels (-1 for noise/outliers)
        """
        min_size = min_cluster_size or self.min_cluster_size

        print(f"\n🎯 Clustering characters (min_cluster_size={min_size})...")

        # Normalize features (ArcFace embeddings are already normalized, but ensure it)
        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

        # HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=min_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(features_norm)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"✓ Found {n_clusters} character clusters")
        print(f"  Noise/outliers: {n_noise} faces")

        # Print cluster sizes
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        print("\n📊 Cluster distribution:")
        for label, count in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
            print(f"  Character {label:2d}: {count:4d} faces")

        return labels

    def visualize_clusters(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        output_path: Path
    ):
        """
        Visualize clusters using PCA

        Args:
            features: Feature vectors
            labels: Cluster labels
            output_path: Path to save visualization
        """
        print("\n📈 Creating cluster visualization...")

        # PCA to 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # Plot
        plt.figure(figsize=(14, 10))

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise: gray
                color = 'gray'
                marker = 'x'
                label_name = 'Noise'
            else:
                marker = 'o'
                label_name = f'Character {label}'

            mask = labels == label
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[color],
                label=label_name,
                marker=marker,
                alpha=0.6,
                s=50
            )

        plt.title('Character Clustering Visualization (PCA)\nInsightFace ArcFace Features', fontsize=14, fontweight='bold')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualization saved to {output_path}")

    def organize_by_cluster(
        self,
        image_paths: List[Path],
        labels: np.ndarray,
        output_dir: Path,
        copy_files: bool = True,
        face_bboxes: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Organize images into cluster-specific folders

        Args:
            image_paths: List of image paths
            labels: Cluster labels
            output_dir: Output directory
            copy_files: If True, copy files; if False, create hard links
            face_bboxes: Optional list of face bounding boxes

        Returns:
            Organization summary
        """
        print(f"\n📁 Organizing images into {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        organization = {
            "total_images": len(image_paths),
            "clusters": {},
            "noise": 0
        }

        # Save face bboxes if provided
        face_data = {}

        for idx, (img_path, label) in enumerate(tqdm(zip(image_paths, labels), total=len(image_paths), desc="Organizing")):
            if label == -1:
                # Noise/outliers
                cluster_dir = output_dir / "noise"
                organization["noise"] += 1
            else:
                cluster_dir = output_dir / f"character_{label:03d}"
                if label not in organization["clusters"]:
                    organization["clusters"][label] = 0
                organization["clusters"][label] += 1

            cluster_dir.mkdir(exist_ok=True)

            dst_path = cluster_dir / img_path.name

            if copy_files:
                shutil.copy2(img_path, dst_path)
            else:
                # Use hard links (Windows-compatible, no extra space)
                if not dst_path.exists():
                    import os
                    try:
                        os.link(img_path, dst_path)
                    except:
                        # Fallback to copy if hard link fails
                        shutil.copy2(img_path, dst_path)

            # Store face bbox info
            if face_bboxes is not None:
                face_data[str(dst_path)] = {
                    "bbox": face_bboxes[idx].tolist(),
                    "label": int(label)
                }

        # Save face data
        if face_bboxes is not None:
            face_data_path = output_dir / "face_detections.json"
            with open(face_data_path, 'w', encoding='utf-8') as f:
                json.dump(face_data, f, indent=2)
            print(f"✓ Face detection data saved: {face_data_path}")

        # Save organization summary
        summary_path = output_dir / "clustering_summary.json"

        # Convert numpy int64 keys to Python int for JSON serialization
        organization_clean = {
            "total_images": organization["total_images"],
            "clusters": {int(k): v for k, v in organization["clusters"].items()},
            "noise": organization["noise"]
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "organization": organization_clean,
                "min_cluster_size": self.min_cluster_size,
                "method": "InsightFace (RetinaFace + ArcFace)"
            }, f, indent=2)

        print(f"✓ Organization complete:")
        print(f"  Characters: {len(organization['clusters'])}")
        print(f"  Noise: {organization['noise']}")
        print(f"  Summary: {summary_path}")

        return organization


def process_layered_frames(
    input_dir: Path,
    output_dir: Path,
    min_cluster_size: int = 10,
    device: str = "cuda",
    copy_files: bool = False,
    visualize: bool = True,
    det_size: Tuple[int, int] = (640, 640)
):
    """
    Process all character layers from segmented frames using AI models

    Args:
        input_dir: Directory with layered frames (episode_XXX/character/)
        output_dir: Output directory for clustered characters
        min_cluster_size: Minimum frames per character
        device: Device to use ('cuda' or 'cpu')
        copy_files: Copy files instead of hard linking
        visualize: Create visualization
        det_size: Detection input size
    """
    print(f"\n{'='*80}")
    print("AI-POWERED CHARACTER CLUSTERING PIPELINE")
    print("Fully GPU-Accelerated: RetinaFace + ArcFace")
    print(f"{'='*80}\n")

    # Find all character images
    print("🔍 Scanning for character images...")
    character_dirs = list(input_dir.glob("*/character"))

    if not character_dirs:
        print(f"❌ No character directories found in {input_dir}")
        return

    all_images = []
    for char_dir in character_dirs:
        images = list(char_dir.glob("*.png"))
        all_images.extend(images)
        print(f"  {char_dir.parent.name}: {len(images)} images")

    print(f"\n✓ Found {len(all_images)} total character images")

    if not all_images:
        print("❌ No images found!")
        return

    # Initialize clusterer
    quality_filter = QualityFilter(
        min_size=64,
        max_blur_threshold=100.0,
        min_alpha_coverage=0.05,
        min_face_size=32
    )

    clusterer = AICharacterClusterer(
        device=device,
        min_cluster_size=min_cluster_size,
        quality_filter=quality_filter,
        det_size=det_size
    )

    # Detect faces and extract features (all GPU-accelerated)
    features, valid_paths, quality_reports, face_bboxes = clusterer.detect_and_extract_features(all_images)

    # Save quality report
    quality_report_path = output_dir / "quality_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(quality_report_path, 'w', encoding='utf-8') as f:
        json.dump(quality_reports, f, indent=2)
    print(f"\n✓ Quality report saved: {quality_report_path}")

    # Cluster
    labels = clusterer.cluster_characters(features)

    # Visualize
    if visualize:
        viz_path = output_dir / "cluster_visualization.png"
        clusterer.visualize_clusters(features, labels, viz_path)

    # Organize
    organization = clusterer.organize_by_cluster(
        valid_paths,
        labels,
        output_dir,
        copy_files=copy_files,
        face_bboxes=face_bboxes
    )

    print(f"\n{'='*80}")
    print("✅ AI CHARACTER CLUSTERING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Output directory: {output_dir}")
    print(f"  Character clusters: {len(organization['clusters'])}")
    print(f"  Total organized images: {organization['total_images']}")
    print(f"  Noise/outliers: {organization['noise']}")
    print(f"\nMethod: InsightFace (RetinaFace detection + ArcFace features)")
    print(f"Device: {device}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Character Clustering Tool (InsightFace)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cluster all characters using AI face detection and features
  python ai_character_clustering.py /path/to/layered_frames --output-dir /path/to/clustered

  # Use larger minimum cluster size
  python ai_character_clustering.py /path/to/layered_frames -o /path/to/output --min-cluster-size 25

  # Run on CPU (slower but no GPU required)
  python ai_character_clustering.py /path/to/layered_frames -o /path/to/output --device cpu

  # Copy files instead of hard linking
  python ai_character_clustering.py /path/to/layered_frames -o /path/to/output --copy
"""
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing layered frames (episode_XXX/character/)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Output directory for clustered characters"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum number of frames per character cluster (default: 10)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating hard links"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip cluster visualization"
    )
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Detection input size (default: 640 640)"
    )

    args = parser.parse_args()

    # Check input directory
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    # Process
    process_layered_frames(
        args.input_dir,
        args.output_dir,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        copy_files=args.copy,
        visualize=not args.no_visualize,
        det_size=tuple(args.det_size)
    )


if __name__ == "__main__":
    main()
