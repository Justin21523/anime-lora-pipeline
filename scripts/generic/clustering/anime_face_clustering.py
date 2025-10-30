#!/usr/bin/env python3
"""
Anime Face-Based Character Clustering Tool
Identifies and clusters anime characters based on facial features

Features:
- Anime-specific face detection (supports multiple detectors)
- Face feature extraction using deep learning
- HDBSCAN clustering for character grouping
- Handles various angles, expressions, and poses
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
import cv2
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import face detection libraries
ANIME_FACE_DETECTOR_AVAILABLE = False
INSIGHTFACE_AVAILABLE = False
OPENCV_CASCADE_AVAILABLE = False

try:
    from anime_face_detector import create_detector
    ANIME_FACE_DETECTOR_AVAILABLE = True
except ImportError:
    pass

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

# Check for OpenCV anime face cascade
OPENCV_CASCADE_PATH = Path(__file__).parent / "lbpcascade_animeface.xml"
if OPENCV_CASCADE_PATH.exists():
    OPENCV_CASCADE_AVAILABLE = True


class AnimeFaceDetector:
    """Wrapper for multiple anime face detection methods"""

    def __init__(self, method: str = "auto", device: str = "cuda"):
        """
        Initialize anime face detector

        Args:
            method: Detection method ("anime-face-detector", "opencv", "auto")
            device: Device for computation
        """
        self.device = device
        self.method = method
        self.detector = None

        if method == "auto":
            # Try methods in order of preference
            if ANIME_FACE_DETECTOR_AVAILABLE:
                self.method = "anime-face-detector"
            elif OPENCV_CASCADE_AVAILABLE:
                self.method = "opencv"
            else:
                raise RuntimeError(
                    "No anime face detector available. Please install:\n"
                    "  pip install anime-face-detector\n"
                    "Or download lbpcascade_animeface.xml"
                )

        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the selected detector"""
        print(f"🔧 Initializing anime face detector: {self.method}")

        if self.method == "anime-face-detector":
            if not ANIME_FACE_DETECTOR_AVAILABLE:
                raise RuntimeError("anime-face-detector not available")
            self.detector = create_detector('yolov3', device=self.device)
            print("✓ YOLOv3-based anime face detector loaded")

        elif self.method == "opencv":
            if not OPENCV_CASCADE_AVAILABLE:
                raise RuntimeError(f"OpenCV cascade not found at {OPENCV_CASCADE_PATH}")
            self.detector = cv2.CascadeClassifier(str(OPENCV_CASCADE_PATH))
            print("✓ OpenCV anime face cascade loaded")

        else:
            raise ValueError(f"Unknown detection method: {self.method}")

    def detect_faces(self, image_path: Path) -> List[Dict]:
        """
        Detect anime faces in image

        Args:
            image_path: Path to image

        Returns:
            List of face detections with bounding boxes
        """
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        if self.method == "anime-face-detector":
            return self._detect_yolo(img_np)
        elif self.method == "opencv":
            return self._detect_opencv(img_np)

    def _detect_yolo(self, img_np: np.ndarray) -> List[Dict]:
        """Detect faces using YOLOv3"""
        preds = self.detector(img_np)

        faces = []
        for pred in preds:
            bbox = pred['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Ensure bbox is valid
            if x2 > x1 and y2 > y1:
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(pred['score'])
                })

        return faces

    def _detect_opencv(self, img_np: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV cascade"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )

        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 1.0  # OpenCV doesn't provide confidence
            })

        return results


class FaceFeatureExtractor:
    """Extract features from detected faces"""

    def __init__(self, method: str = "resnet", device: str = "cuda"):
        """
        Initialize feature extractor

        Args:
            method: Feature extraction method ("resnet", "insightface")
            device: Device for computation
        """
        self.device = device
        self.method = method

        if method == "insightface" and not INSIGHTFACE_AVAILABLE:
            print("⚠️  InsightFace not available, falling back to ResNet")
            self.method = "resnet"

        self._initialize_model()

    def _initialize_model(self):
        """Initialize feature extraction model"""
        if self.method == "insightface":
            print("🔧 Loading InsightFace model...")
            self.model = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.model.prepare(ctx_id=0 if self.device == "cuda" else -1)
            print("✓ InsightFace loaded")

        elif self.method == "resnet":
            print("🔧 Loading ResNet50 feature extractor...")
            from torchvision import models, transforms

            self.model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(self.device)

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            print("✓ ResNet50 loaded")

    def extract_face_features(
        self,
        img: np.ndarray,
        bbox: List[int]
    ) -> Optional[np.ndarray]:
        """
        Extract features from a face region

        Args:
            img: Full image as numpy array
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Feature vector or None if extraction fails
        """
        x1, y1, x2, y2 = bbox

        # Expand bbox slightly to include more context
        h, w = img.shape[:2]
        margin = 0.1
        x_margin = int((x2 - x1) * margin)
        y_margin = int((y2 - y1) * margin)

        x1 = max(0, x1 - x_margin)
        y1 = max(0, y1 - y_margin)
        x2 = min(w, x2 + x_margin)
        y2 = min(h, y2 + y_margin)

        face_img = img[y1:y2, x1:x2]

        if face_img.size == 0:
            return None

        if self.method == "resnet":
            return self._extract_resnet(face_img)
        elif self.method == "insightface":
            return self._extract_insightface(face_img)

    def _extract_resnet(self, face_img: np.ndarray) -> np.ndarray:
        """Extract features using ResNet"""
        face_pil = Image.fromarray(face_img)
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(face_tensor)
            features = features.squeeze().cpu().numpy()

        return features

    def _extract_insightface(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract features using InsightFace"""
        faces = self.model.get(face_img)
        if len(faces) > 0:
            return faces[0].embedding
        return None


class AnimeFaceClusterer:
    """Main anime face clustering pipeline"""

    def __init__(
        self,
        detection_method: str = "auto",
        feature_method: str = "resnet",
        device: str = "cuda",
        min_cluster_size: int = 10,
        min_confidence: float = 0.3
    ):
        """
        Initialize clustering pipeline

        Args:
            detection_method: Face detection method
            feature_method: Feature extraction method
            device: Device for computation
            min_cluster_size: Minimum faces per character cluster
            min_confidence: Minimum face detection confidence
        """
        self.device = device
        self.min_cluster_size = min_cluster_size
        self.min_confidence = min_confidence

        self.face_detector = AnimeFaceDetector(detection_method, device)
        self.feature_extractor = FaceFeatureExtractor(feature_method, device)

    def process_images(
        self,
        image_paths: List[Path],
        save_debug: bool = False,
        debug_dir: Optional[Path] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process images to extract face features

        Args:
            image_paths: List of image paths
            save_debug: Save debug images with face detections
            debug_dir: Directory to save debug images

        Returns:
            (feature_matrix, face_info_list)
        """
        print(f"\n🔍 Processing {len(image_paths)} images...")

        features_list = []
        face_info_list = []

        if save_debug and debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(image_paths, desc="Detecting faces"):
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                faces = self.face_detector.detect_faces(img_path)

                # Filter by confidence
                faces = [f for f in faces if f['confidence'] >= self.min_confidence]

                if not faces:
                    continue

                # Process each detected face
                for face_idx, face in enumerate(faces):
                    features = self.feature_extractor.extract_face_features(
                        img, face['bbox']
                    )

                    if features is not None:
                        features_list.append(features)
                        face_info_list.append({
                            'image_path': str(img_path),
                            'bbox': face['bbox'],
                            'confidence': face['confidence'],
                            'face_idx': face_idx
                        })

                # Save debug image
                if save_debug and debug_dir and faces:
                    debug_img = img.copy()
                    for face in faces:
                        x1, y1, x2, y2 = face['bbox']
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        conf_text = f"{face['confidence']:.2f}"
                        cv2.putText(
                            debug_img, conf_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )

                    debug_path = debug_dir / f"debug_{img_path.stem}.jpg"
                    cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"\n⚠️  Failed to process {img_path.name}: {e}")
                continue

        if not features_list:
            raise ValueError("No faces detected in any images!")

        features_matrix = np.vstack(features_list)

        print(f"✓ Detected {len(face_info_list)} faces from {len(image_paths)} images")
        print(f"  Feature matrix shape: {features_matrix.shape}")

        return features_matrix, face_info_list

    def cluster_faces(
        self,
        features: np.ndarray,
        min_cluster_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Cluster face features

        Args:
            features: Feature matrix
            min_cluster_size: Minimum cluster size (overrides default)

        Returns:
            Cluster labels (-1 for noise/outliers)
        """
        min_size = min_cluster_size or self.min_cluster_size

        print(f"\n🎯 Clustering faces (min_cluster_size={min_size})...")

        # Normalize features
        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

        # HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=min_size,
            min_samples=3,
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
        """Create cluster visualization"""
        print("\n📈 Creating cluster visualization...")

        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        plt.figure(figsize=(14, 10))

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'gray'
                marker = 'x'
                label_name = 'Noise'
                alpha = 0.3
            else:
                marker = 'o'
                label_name = f'Character {label}'
                alpha = 0.6

            mask = labels == label
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[color],
                label=label_name,
                marker=marker,
                alpha=alpha,
                s=50
            )

        plt.title('Anime Character Face Clustering (PCA)', fontsize=14)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualization saved to {output_path}")

    def organize_by_cluster(
        self,
        face_info_list: List[Dict],
        labels: np.ndarray,
        output_dir: Path,
        copy_mode: str = "link"
    ) -> Dict:
        """
        Organize images by character cluster

        Args:
            face_info_list: List of face information
            labels: Cluster labels
            output_dir: Output directory
            copy_mode: "link" (hard link), "copy" (copy files), or "info" (JSON only)

        Returns:
            Organization summary
        """
        print(f"\n📁 Organizing faces into {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        organization = {
            "total_faces": len(face_info_list),
            "characters": {},
            "noise": 0
        }

        for face_info, label in tqdm(
            zip(face_info_list, labels),
            total=len(face_info_list),
            desc="Organizing"
        ):
            img_path = Path(face_info['image_path'])

            if label == -1:
                cluster_dir = output_dir / "noise"
                organization["noise"] += 1
            else:
                cluster_dir = output_dir / f"character_{label:03d}"
                if label not in organization["characters"]:
                    organization["characters"][int(label)] = {
                        "count": 0,
                        "images": []
                    }
                organization["characters"][int(label)]["count"] += 1
                organization["characters"][int(label)]["images"].append({
                    "path": str(img_path),
                    "bbox": face_info['bbox'],
                    "confidence": face_info['confidence']
                })

            if copy_mode in ["link", "copy"]:
                cluster_dir.mkdir(exist_ok=True)

                # Create unique filename
                face_idx = face_info.get('face_idx', 0)
                dst_name = f"{img_path.stem}_face{face_idx}{img_path.suffix}"
                dst_path = cluster_dir / dst_name

                if copy_mode == "link":
                    if not dst_path.exists():
                        import os
                        try:
                            os.link(img_path, dst_path)
                        except OSError:
                            # Hard link failed, copy instead
                            import shutil
                            shutil.copy2(img_path, dst_path)
                elif copy_mode == "copy":
                    import shutil
                    shutil.copy2(img_path, dst_path)

        # Save organization summary
        summary_path = output_dir / "face_clustering_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "organization": organization,
                "min_cluster_size": self.min_cluster_size,
                "min_confidence": self.min_confidence
            }, f, indent=2, ensure_ascii=False)

        print(f"✓ Organization complete:")
        print(f"  Characters: {len(organization['characters'])}")
        print(f"  Noise: {organization['noise']}")
        print(f"  Summary: {summary_path}")

        return organization


def main():
    parser = argparse.ArgumentParser(
        description="Anime Face-Based Character Clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cluster using default settings
  python anime_face_clustering.py /path/to/layered_frames -o /path/to/output

  # Use specific detection method and higher confidence threshold
  python anime_face_clustering.py /path/to/images -o output \\
      --detection-method anime-face-detector --min-confidence 0.5

  # Save debug images to inspect face detections
  python anime_face_clustering.py /path/to/images -o output --save-debug
"""
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing character images"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Output directory for clustered characters"
    )
    parser.add_argument(
        "--detection-method",
        choices=["auto", "anime-face-detector", "opencv"],
        default="auto",
        help="Face detection method (default: auto)"
    )
    parser.add_argument(
        "--feature-method",
        choices=["resnet", "insightface"],
        default="resnet",
        help="Feature extraction method (default: resnet)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum faces per character cluster (default: 10)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum face detection confidence (default: 0.3)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--copy-mode",
        choices=["link", "copy", "info"],
        default="link",
        help="File organization mode: link (hard link), copy (copy files), info (JSON only)"
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug images with face detections"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip cluster visualization"
    )

    args = parser.parse_args()

    # Check input
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    print(f"\n{'='*80}")
    print("ANIME FACE-BASED CHARACTER CLUSTERING")
    print(f"{'='*80}\n")

    # Find all images
    print("🔍 Scanning for character images...")
    all_images = []

    # Check if input is layered frames directory structure
    character_dirs = list(args.input_dir.glob("*/character"))
    if character_dirs:
        for char_dir in character_dirs:
            images = list(char_dir.glob("*.png"))
            all_images.extend(images)
            print(f"  {char_dir.parent.name}: {len(images)} images")
    else:
        # Scan recursively for all image files
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            all_images.extend(args.input_dir.rglob(ext))
        print(f"  Found {len(all_images)} images")

    if not all_images:
        print("❌ No images found!")
        return

    print(f"\n✓ Total images to process: {len(all_images)}")

    # Initialize clusterer
    clusterer = AnimeFaceClusterer(
        detection_method=args.detection_method,
        feature_method=args.feature_method,
        device=args.device,
        min_cluster_size=args.min_cluster_size,
        min_confidence=args.min_confidence
    )

    # Process images
    debug_dir = args.output_dir / "debug_detections" if args.save_debug else None
    features, face_info = clusterer.process_images(
        all_images,
        save_debug=args.save_debug,
        debug_dir=debug_dir
    )

    # Cluster
    labels = clusterer.cluster_faces(features)

    # Visualize
    if not args.no_visualize:
        viz_path = args.output_dir / "face_cluster_visualization.png"
        clusterer.visualize_clusters(features, labels, viz_path)

    # Organize
    organization = clusterer.organize_by_cluster(
        face_info,
        labels,
        args.output_dir,
        copy_mode=args.copy_mode
    )

    print(f"\n{'='*80}")
    print("✅ FACE-BASED CHARACTER CLUSTERING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Output directory: {args.output_dir}")
    print(f"  Character clusters: {len(organization['characters'])}")
    print(f"  Total faces: {organization['total_faces']}")
    print(f"  Noise/outliers: {organization['noise']}")
    print()


if __name__ == "__main__":
    main()
