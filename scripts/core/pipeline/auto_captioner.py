"""
Auto Captioner using WD14 Tagger v3 for automatic image tagging
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import pandas as pd
from huggingface_hub import hf_hub_download
import onnxruntime as ort

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.logger import get_logger, create_progress_bar, print_success, print_error, print_section
from core.utils.image_utils import load_image_rgb
from core.utils.path_utils import ensure_dir, list_images


logger = get_logger("AutoCaptioner")


class WD14Tagger:
    """
    WD14 Tagger v3 wrapper for anime image tagging
    """

    def __init__(
        self,
        model_repo: str = "SmilingWolf/wd-vit-tagger-v3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize WD14 Tagger

        Args:
            model_repo: HuggingFace model repository
            device: Device to run inference on
        """
        self.model_repo = model_repo
        self.device = device
        self.model = None
        self.tags = None
        self.general_threshold = 0.35
        self.character_threshold = 0.85

        logger.info(f"Initializing WD14 Tagger from {model_repo}")
        self._load_model()

    def _load_model(self):
        """Load ONNX model and tag list"""
        try:
            # Download model and tags
            model_path = hf_hub_download(
                self.model_repo,
                filename="model.onnx"
            )

            tags_path = hf_hub_download(
                self.model_repo,
                filename="selected_tags.csv"
            )

            # Load ONNX model
            if self.device == "cuda" and torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            self.model = ort.InferenceSession(model_path, providers=providers)

            # Load tags
            self.tags = pd.read_csv(tags_path)
            logger.success(f"Loaded {len(self.tags)} tags")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _prepare_image(self, image: Image.Image, target_size: int = 448) -> np.ndarray:
        """
        Prepare image for model input

        Args:
            image: PIL Image
            target_size: Target size for model input

        Returns:
            Preprocessed image array
        """
        # Resize and pad
        image = image.convert('RGB')
        width, height = image.size

        # Calculate padding
        max_dim = max(width, height)
        pad_left = (max_dim - width) // 2
        pad_top = (max_dim - height) // 2

        # Create padded image
        padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))

        # Resize to target size
        padded = padded.resize((target_size, target_size), Image.BICUBIC)

        # Convert to array and normalize
        img_array = np.array(padded, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1]

        # Add batch dimension (WD14 expects batch, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(
        self,
        image_path: Path,
        general_threshold: Optional[float] = None,
        character_threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Predict tags for an image

        Args:
            image_path: Path to image file
            general_threshold: Threshold for general tags
            character_threshold: Threshold for character tags

        Returns:
            Dictionary of {tag: confidence}
        """
        general_threshold = general_threshold or self.general_threshold
        character_threshold = character_threshold or self.character_threshold

        try:
            # Load and prepare image
            image = Image.open(image_path)
            input_array = self._prepare_image(image)

            # Run inference
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_array})[0]

            # Get predictions
            predictions = output[0]

            # Filter by threshold
            results = {}
            for i, tag_data in self.tags.iterrows():
                confidence = float(predictions[i])
                tag_name = tag_data['name']
                tag_category = tag_data['category']

                # Apply different thresholds
                threshold = character_threshold if tag_category == 4 else general_threshold

                if confidence >= threshold:
                    results[tag_name] = confidence

            return results

        except Exception as e:
            logger.error(f"Error predicting tags for {image_path}: {e}")
            return {}


class AutoCaptioner:
    """
    Automatic caption generator for LoRA training
    """

    def __init__(
        self,
        model_repo: str = "SmilingWolf/wd-vit-tagger-v3",
        trigger_word: str = "",
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize AutoCaptioner

        Args:
            model_repo: WD14 model repository
            trigger_word: Character trigger word (e.g., "endou_mamoru")
            general_threshold: Threshold for general tags
            character_threshold: Threshold for character tags
            device: Device for inference
        """
        self.trigger_word = trigger_word
        self.tagger = WD14Tagger(model_repo, device)
        self.tagger.general_threshold = general_threshold
        self.tagger.character_threshold = character_threshold

    def generate_caption(
        self,
        image_path: Path,
        style: str = "detailed",
        tag_blacklist: Optional[List[str]] = None
    ) -> str:
        """
        Generate caption for a single image

        Args:
            image_path: Path to image
            style: Caption style ("minimal" or "detailed")
            tag_blacklist: Tags to exclude

        Returns:
            Caption string
        """
        tag_blacklist = tag_blacklist or ["solo", "1boy", "1girl", "anime_coloring"]

        # Get tags
        tags = self.tagger.predict(image_path)

        # Filter blacklisted tags
        filtered_tags = {
            tag: conf for tag, conf in tags.items()
            if tag not in tag_blacklist
        }

        # Sort by confidence
        sorted_tags = sorted(
            filtered_tags.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if style == "minimal":
            # Minimal style: just trigger word
            return self.trigger_word if self.trigger_word else ""

        elif style == "detailed":
            # Detailed style: trigger word + top tags
            tag_strings = [tag for tag, _ in sorted_tags[:30]]  # Top 30 tags

            if self.trigger_word:
                caption = f"{self.trigger_word}, " + ", ".join(tag_strings)
            else:
                caption = ", ".join(tag_strings)

            return caption

        else:
            raise ValueError(f"Unknown style: {style}")

    def batch_caption(
        self,
        image_dir: Path,
        output_dir: Optional[Path] = None,
        style: str = "detailed",
        tag_blacklist: Optional[List[str]] = None,
        save_individual_files: bool = True,
        save_combined_file: bool = False
    ) -> Dict[Path, str]:
        """
        Generate captions for all images in directory

        Args:
            image_dir: Directory containing images
            output_dir: Output directory (default: same as image_dir)
            style: Caption style
            tag_blacklist: Tags to exclude
            save_individual_files: Save .txt files for each image
            save_combined_file: Save combined metadata.jsonl

        Returns:
            Dictionary mapping image paths to captions
        """
        output_dir = output_dir or image_dir
        ensure_dir(output_dir)

        # Get all images
        image_files = list_images(image_dir, recursive=False)
        logger.info(f"Found {len(image_files)} images to caption")

        captions = {}

        print_section(f"Generating Captions ({style} style)")

        with create_progress_bar() as progress:
            task = progress.add_task(
                f"[cyan]Processing images...",
                total=len(image_files)
            )

            for image_path in image_files:
                try:
                    # Generate caption
                    caption = self.generate_caption(
                        image_path,
                        style=style,
                        tag_blacklist=tag_blacklist
                    )

                    captions[image_path] = caption

                    # Save individual caption file
                    if save_individual_files:
                        caption_file = output_dir / f"{image_path.stem}.txt"
                        caption_file.write_text(caption, encoding='utf-8')

                except Exception as e:
                    logger.error(f"Error processing {image_path.name}: {e}")

                progress.update(task, advance=1)

        # Save combined metadata file
        if save_combined_file:
            metadata_file = output_dir / "captions_metadata.jsonl"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                for img_path, caption in captions.items():
                    import json
                    entry = {
                        "image": img_path.name,
                        "caption": caption
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print_success(f"Generated {len(captions)} captions")
        return captions


def main():
    """Test AutoCaptioner"""
    import argparse

    parser = argparse.ArgumentParser(description="Auto Caption Images")
    parser.add_argument("--image_dir", type=Path, required=True, help="Image directory")
    parser.add_argument("--trigger_word", type=str, default="", help="Trigger word")
    parser.add_argument("--style", type=str, default="detailed", choices=["minimal", "detailed"])
    parser.add_argument("--output_dir", type=Path, help="Output directory")

    args = parser.parse_args()

    captioner = AutoCaptioner(
        trigger_word=args.trigger_word,
        general_threshold=0.35
    )

    captions = captioner.batch_caption(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        style=args.style,
        save_individual_files=True,
        save_combined_file=True
    )

    print(f"\n✓ Captioned {len(captions)} images")


if __name__ == "__main__":
    main()
