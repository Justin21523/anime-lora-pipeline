#!/usr/bin/env python3
"""
Stage 1: YOLOv8-seg Instance Segmentation
Alternative to Detectron2/Mask2Former for character/object detection and segmentation
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import logging
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv8Segmenter:
    """
    YOLOv8-seg wrapper for instance segmentation
    Fast, efficient alternative to Mask2Former
    """

    def __init__(
        self,
        model_size: str = "x",  # n, s, m, l, x
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 100
    ):
        """
        Initialize YOLOv8-seg model

        Args:
            model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            device: Device to run on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_det: Maximum detections per image
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        # Load YOLOv8-seg model
        model_name = f"yolov8{model_size}-seg.pt"
        logger.info(f"Loading YOLOv8-seg model: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)

        # COCO class names (YOLOv8 is pretrained on COCO)
        self.coco_classes = self.model.names

        # Classes relevant for anime character detection
        # COCO: 0=person, 27=tie, 28=suitcase, 31=backpack, etc.
        self.character_classes = [0]  # person
        self.object_classes = list(range(1, 80))  # all other objects

        logger.info(f"YOLOv8-seg initialized on {device}")
        logger.info(f"Available classes: {len(self.coco_classes)}")

    @torch.no_grad()
    def predict(
        self,
        images: List[str],
        batch_size: int = 8,
        imgsz: int = 640,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Run inference on a batch of images

        Args:
            images: List of image paths
            batch_size: Batch size for processing
            imgsz: Input image size (will be resized)
            verbose: Whether to show prediction details

        Returns:
            List of prediction dictionaries with masks, boxes, classes, scores
        """
        results = []

        # Process in batches
        for i in tqdm(range(0, len(images), batch_size), desc="YOLOv8 Inference"):
            batch_paths = images[i:i + batch_size]

            # Run inference
            batch_results = self.model.predict(
                source=batch_paths,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                imgsz=imgsz,
                device=self.device,
                verbose=verbose,
                save=False,
                stream=False
            )

            # Parse results
            for result in batch_results:
                parsed = self._parse_result(result)
                results.append(parsed)

        return results

    def _parse_result(self, result) -> Dict:
        """
        Parse YOLOv8 result into standard format

        Returns:
            Dict with:
                - masks: List of binary masks (H, W) np.arrays
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - classes: List of class IDs
                - class_names: List of class names
                - scores: List of confidence scores
                - image_shape: (H, W) of original image
        """
        parsed = {
            'masks': [],
            'boxes': [],
            'classes': [],
            'class_names': [],
            'scores': [],
            'image_shape': result.orig_shape  # (H, W)
        }

        # Check if segmentation masks exist
        if result.masks is None:
            return parsed

        # Extract masks (already in original image size)
        masks = result.masks.data.cpu().numpy()  # (N, H, W)

        # Extract boxes
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) - [x1, y1, x2, y2]

        # Extract classes and scores
        classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        scores = result.boxes.conf.cpu().numpy()  # (N,)

        # Convert to lists
        for mask, box, cls, score in zip(masks, boxes, classes, scores):
            parsed['masks'].append(mask)
            parsed['boxes'].append(box.tolist())
            parsed['classes'].append(int(cls))
            parsed['class_names'].append(self.coco_classes[int(cls)])
            parsed['scores'].append(float(score))

        return parsed

    def extract_character_masks(
        self,
        prediction: Dict,
        min_area: int = 1000,
        combine_overlapping: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract character masks from prediction

        Args:
            prediction: Parsed prediction dictionary
            min_area: Minimum mask area in pixels
            combine_overlapping: Whether to combine overlapping masks

        Returns:
            - combined_mask: Single binary mask (H, W) with all characters
            - character_info: List of dicts with mask, box, score for each character
        """
        H, W = prediction['image_shape']
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        character_info = []

        # Filter for character class (person=0 in COCO)
        for i, cls in enumerate(prediction['classes']):
            if cls not in self.character_classes:
                continue

            mask = prediction['masks'][i]

            # Resize mask to match image shape if needed
            if mask.shape != (H, W):
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((W, H), PILImage.Resampling.BILINEAR)
                mask = np.array(mask_img) / 255.0

            # Check minimum area
            if mask.sum() < min_area:
                continue

            # Add to combined mask
            mask_binary = (mask > 0.5).astype(np.uint8)
            if combine_overlapping:
                combined_mask = np.maximum(combined_mask, mask_binary)
            else:
                combined_mask[(mask > 0.5) & (combined_mask == 0)] = 1

            # Store character info
            character_info.append({
                'mask': mask_binary,
                'box': prediction['boxes'][i],
                'score': prediction['scores'][i],
                'class_name': prediction['class_names'][i]
            })

        return combined_mask, character_info

    def save_masks(
        self,
        image_path: str,
        prediction: Dict,
        output_dir: Path,
        save_combined: bool = True,
        save_individual: bool = False,
        save_visualization: bool = True
    ):
        """
        Save segmentation masks to disk

        Args:
            image_path: Path to original image
            prediction: Parsed prediction dictionary
            output_dir: Output directory
            save_combined: Save combined character mask
            save_individual: Save individual instance masks
            save_visualization: Save visualization overlay
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get filename
        img_name = Path(image_path).stem

        # Extract character masks
        combined_mask, char_info = self.extract_character_masks(prediction)

        # Save combined mask
        if save_combined and combined_mask.max() > 0:
            mask_path = output_dir / f"{img_name}_character_mask.png"
            Image.fromarray((combined_mask * 255).astype(np.uint8)).save(mask_path)

        # Save individual masks
        if save_individual:
            for idx, info in enumerate(char_info):
                mask_path = output_dir / f"{img_name}_char_{idx:02d}.png"
                Image.fromarray((info['mask'] * 255).astype(np.uint8)).save(mask_path)

        # Save visualization
        if save_visualization and len(char_info) > 0:
            # Load original image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)

            # Create overlay
            overlay = img_array.copy()

            # Color each mask differently
            colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
            ]

            for idx, info in enumerate(char_info):
                color = colors[idx % len(colors)]
                mask = info['mask']

                # Apply colored overlay
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        mask > 0,
                        overlay[:, :, c] * 0.5 + color[c] * 0.5,
                        overlay[:, :, c]
                    )

            # Save
            vis_path = output_dir / f"{img_name}_visualization.png"
            Image.fromarray(overlay.astype(np.uint8)).save(vis_path)

        return combined_mask, char_info


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    model_size: str = "x",
    batch_size: int = 8,
    conf_threshold: float = 0.25,
    device: str = "cuda"
):
    """
    Process entire dataset with YOLOv8-seg

    Args:
        input_dir: Directory containing images
        output_dir: Output directory for masks
        model_size: YOLOv8 model size
        batch_size: Batch size for inference
        conf_threshold: Confidence threshold
        device: Device to run on
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    segmenter = YOLOv8Segmenter(
        model_size=model_size,
        device=device,
        conf_threshold=conf_threshold
    )

    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg']
    images = []
    for ext in image_extensions:
        images.extend(list(input_dir.rglob(f'*{ext}')))

    logger.info(f"Found {len(images)} images in {input_dir}")

    # Convert to strings
    image_paths = [str(p) for p in images]

    # Run inference
    predictions = segmenter.predict(
        images=image_paths,
        batch_size=batch_size,
        verbose=False
    )

    # Save results
    logger.info("Saving masks...")
    stats = {
        'total_images': len(images),
        'images_with_characters': 0,
        'total_characters': 0
    }

    for img_path, pred in tqdm(zip(image_paths, predictions), total=len(images)):
        combined_mask, char_info = segmenter.save_masks(
            image_path=img_path,
            prediction=pred,
            output_dir=output_dir,
            save_combined=True,
            save_individual=False,
            save_visualization=True
        )

        if len(char_info) > 0:
            stats['images_with_characters'] += 1
            stats['total_characters'] += len(char_info)

    logger.info(f"Processing complete!")
    logger.info(f"Images with characters: {stats['images_with_characters']}/{stats['total_images']}")
    logger.info(f"Total characters detected: {stats['total_characters']}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8-seg Stage 1 Segmentation")
    parser.add_argument("input_dir", type=str, help="Input directory with images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model-size", type=str, default="x", choices=['n', 's', 'm', 'l', 'x'],
                        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    process_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_size=args.model_size,
        batch_size=args.batch_size,
        conf_threshold=args.conf_threshold,
        device=args.device
    )
