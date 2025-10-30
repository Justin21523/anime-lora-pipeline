#!/usr/bin/env python3
"""
Stage 1: Base Segmentation with Mask2Former
Performs semantic, instance, and panoptic segmentation on anime frames
Outputs coarse masks for characters, objects, and background
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

try:
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    # Try to import Mask2Former
    try:
        from mask2former import add_maskformer2_config
        MASK2FORMER_AVAILABLE = True
    except ImportError:
        MASK2FORMER_AVAILABLE = False
        print("⚠️  Mask2Former not installed. Install with: pip install git+https://github.com/facebookresearch/Mask2Former.git")
except ImportError:
    MASK2FORMER_AVAILABLE = False
    print("⚠️  Detectron2 not installed. Install with: pip install detectron2")


class AnimeFrameDataset(Dataset):
    """Dataset for anime frame loading with efficient batch processing"""
    
    def __init__(self, frame_paths: List[Path]):
        """
        Args:
            frame_paths: List of paths to frame images
        """
        self.frame_paths = frame_paths
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        
        try:
            # Load image
            image = cv2.imread(str(frame_path))
            if image is None:
                raise ValueError(f"Failed to load image: {frame_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return {
                'image': image,
                'path': str(frame_path),
                'filename': frame_path.name
            }
        except Exception as e:
            print(f"Error loading {frame_path}: {e}")
            # Return dummy data
            return {
                'image': np.zeros((512, 512, 3), dtype=np.uint8),
                'path': str(frame_path),
                'filename': frame_path.name
            }


class Mask2FormerSegmenter:
    """
    Mask2Former-based segmentation for anime frames
    Supports semantic, instance, and panoptic segmentation
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        weights_file: Optional[str] = None,
        device: str = 'cuda',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize Mask2Former segmenter
        
        Args:
            config_file: Path to Mask2Former config file
            weights_file: Path to pretrained weights
            device: Device to run inference on
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        if not MASK2FORMER_AVAILABLE:
            raise ImportError("Mask2Former not available. Please install dependencies.")
        
        print("🔄 Loading Mask2Former model...")
        
        # Setup config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        
        if config_file is None:
            # Use default COCO-trained model config
            config_file = "configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml"
        
        cfg.merge_from_file(config_file)
        
        if weights_file:
            cfg.MODEL.WEIGHTS = weights_file
        
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.DEVICE = device
        
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else "coco_2017_val")
        
        print("✓ Mask2Former model loaded")
    
    def segment_image(self, image: np.ndarray) -> Dict:
        """
        Perform segmentation on a single image
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Dictionary containing:
                - semantic_mask: Semantic segmentation mask
                - instance_masks: List of instance masks
                - panoptic_seg: Panoptic segmentation
                - scores: Confidence scores for instances
                - classes: Class IDs for instances
        """
        # Run inference
        outputs = self.predictor(image)
        
        result = {}
        
        # Extract predictions
        if "sem_seg" in outputs:
            # Semantic segmentation
            sem_seg = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
            result['semantic_mask'] = sem_seg
        
        if "instances" in outputs:
            # Instance segmentation
            instances = outputs["instances"].to("cpu")
            
            masks = instances.pred_masks.numpy() if len(instances) > 0 else []
            scores = instances.scores.numpy() if len(instances) > 0 else []
            classes = instances.pred_classes.numpy() if len(instances) > 0 else []
            
            result['instance_masks'] = masks
            result['scores'] = scores
            result['classes'] = classes
        
        if "panoptic_seg" in outputs:
            # Panoptic segmentation
            panoptic_seg, segments_info = outputs["panoptic_seg"]
            panoptic_seg = panoptic_seg.cpu().numpy()
            
            result['panoptic_mask'] = panoptic_seg
            result['segments_info'] = segments_info
        
        return result
    
    def extract_character_mask(self, result: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract unified character mask from segmentation result
        Combines person class instances into a single mask
        
        Args:
            result: Segmentation result dictionary
            image_shape: (height, width) of original image
            
        Returns:
            Binary mask (H, W) with 255 for character pixels
        """
        h, w = image_shape[:2]
        character_mask = np.zeros((h, w), dtype=np.uint8)
        
        if 'instance_masks' in result and len(result['instance_masks']) > 0:
            # Assume class 0 is 'person' in COCO dataset
            # Adjust this based on your dataset
            person_class_id = 0
            
            for mask, cls in zip(result['instance_masks'], result['classes']):
                if cls == person_class_id:
                    character_mask = np.maximum(character_mask, mask.astype(np.uint8) * 255)
        
        return character_mask
    
    def process_batch(
        self,
        images: List[np.ndarray],
        extract_characters: bool = True
    ) -> List[Dict]:
        """
        Process a batch of images
        
        Args:
            images: List of RGB images
            extract_characters: Whether to extract character masks
            
        Returns:
            List of segmentation results
        """
        results = []
        
        for image in images:
            result = self.segment_image(image)
            
            if extract_characters:
                char_mask = self.extract_character_mask(result, image.shape)
                result['character_mask'] = char_mask
            
            results.append(result)
        
        return results


def process_video_frames(
    input_dir: Path,
    output_dir: Path,
    config_file: Optional[str] = None,
    weights_file: Optional[str] = None,
    batch_size: int = 1,  # Mask2Former processes one at a time
    device: str = 'cuda',
    confidence_threshold: float = 0.5
):
    """
    Process all video frames in a directory
    
    Args:
        input_dir: Directory containing frame images
        output_dir: Directory to save segmentation results
        config_file: Mask2Former config file path
        weights_file: Mask2Former weights path
        batch_size: Batch size (kept at 1 for Mask2Former)
        device: Device for inference
        confidence_threshold: Minimum confidence threshold
    """
    print(f"\n{'='*80}")
    print("🎬 Stage 1: Mask2Former Base Segmentation")
    print(f"{'='*80}\n")
    
    # Create segmenter
    segmenter = Mask2FormerSegmenter(
        config_file=config_file,
        weights_file=weights_file,
        device=device,
        confidence_threshold=confidence_threshold
    )
    
    # Collect all frame paths
    print("📂 Scanning for frames...")
    frame_paths = []
    
    for img_path in sorted(input_dir.rglob('*.jpg')) + sorted(input_dir.rglob('*.png')):
        if not any(skip in img_path.name for skip in ['_mask', '_character', '_background']):
            frame_paths.append(img_path)
    
    print(f"✓ Found {len(frame_paths)} frames\n")
    
    if len(frame_paths) == 0:
        print("⚠️  No frames found!")
        return
    
    # Create dataset and dataloader
    dataset = AnimeFrameDataset(frame_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)
    
    # Process frames
    print("🎨 Performing segmentation...\n")
    
    with tqdm(total=len(frame_paths), desc="Segmenting", unit="frame") as pbar:
        for batch in dataloader:
            image = batch['image'][0].numpy()  # Get single image
            path = batch['path'][0]
            filename = batch['filename'][0]
            
            # Segment
            result = segmenter.segment_image(image)
            
            # Extract character mask
            char_mask = segmenter.extract_character_mask(result, image.shape)
            
            # Save character mask
            mask_path = masks_dir / filename.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
            cv2.imwrite(str(mask_path), char_mask)
            
            # Optionally save other masks
            if 'semantic_mask' in result:
                sem_path = masks_dir / filename.replace('.jpg', '_semantic.png').replace('.png', '_semantic.png')
                # Convert to color for visualization
                sem_vis = (result['semantic_mask'] * 10).astype(np.uint8)
                cv2.imwrite(str(sem_path), sem_vis)
            
            pbar.update(1)
    
    print(f"\n✅ Segmentation complete!")
    print(f"Results saved to: {output_dir}\n")


def main():
    """Main entry point for Stage 1 segmentation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Stage 1: Mask2Former Base Segmentation'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Input directory containing frame images'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        required=True,
        help='Output directory for segmentation results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to Mask2Former config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to Mask2Former weights'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Check input
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run processing
    process_video_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_file=args.config,
        weights_file=args.weights,
        batch_size=args.batch_size,
        device=args.device,
        confidence_threshold=args.confidence
    )


if __name__ == '__main__':
    main()
