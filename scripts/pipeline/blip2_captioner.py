"""
BLIP-2 Captioner for anime character images
More accurate than WD14 for natural language descriptions
"""

import torch
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger, create_progress_bar, print_success, print_error, print_section
from utils.path_utils import ensure_dir, list_images


logger = get_logger("BLIP2Captioner")


class BLIP2Captioner:
    """
    BLIP-2 based image captioner for anime characters
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        trigger_word: str = "",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize BLIP-2 Captioner

        Args:
            model_name: BLIP-2 model to use
            trigger_word: Character trigger word (e.g., "endou_mamoru")
            device: Device for inference
        """
        self.trigger_word = trigger_word
        self.device = device
        self.processor = None
        self.model = None

        logger.info(f"Initializing BLIP-2 from {model_name}")
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load BLIP-2 model and processor"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration

            logger.info("Loading BLIP-2 processor...")
            self.processor = Blip2Processor.from_pretrained(model_name)

            logger.info("Loading BLIP-2 model...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()

            logger.success(f"BLIP-2 loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            raise

    def generate_caption(
        self,
        image_path: Path,
        prompt: str = "a photo of",
        max_length: int = 50,
        num_beams: int = 5
    ) -> str:
        """
        Generate caption for a single image

        Args:
            image_path: Path to image
            prompt: Optional prompt to guide generation
            max_length: Maximum caption length
            num_beams: Number of beams for beam search

        Returns:
            Generated caption string
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Prepare inputs
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(
                self.device, torch.float16 if self.device == "cuda" else torch.float32
            )

            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams
                )

            # Decode
            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            # Add trigger word at the beginning
            if self.trigger_word:
                caption = f"{self.trigger_word}, {caption}"

            return caption

        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return self.trigger_word if self.trigger_word else ""

    def batch_caption(
        self,
        image_dir: Path,
        output_dir: Optional[Path] = None,
        prompt: str = "a photo of",
        save_individual_files: bool = True,
        save_combined_file: bool = False
    ) -> Dict[Path, str]:
        """
        Generate captions for all images in directory

        Args:
            image_dir: Directory containing images
            output_dir: Output directory (default: same as image_dir)
            prompt: Prompt to guide generation
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

        print_section("Generating Captions with BLIP-2")

        with create_progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Processing images...",
                total=len(image_files)
            )

            for image_path in image_files:
                try:
                    # Generate caption
                    caption = self.generate_caption(
                        image_path,
                        prompt=prompt
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
                import json
                for img_path, caption in captions.items():
                    entry = {
                        "image": img_path.name,
                        "caption": caption
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print_success(f"Generated {len(captions)} captions")
        return captions


def main():
    """Test BLIP-2 Captioner"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Captions with BLIP-2")
    parser.add_argument("--image_dir", type=Path, required=True, help="Image directory")
    parser.add_argument("--trigger_word", type=str, default="", help="Trigger word")
    parser.add_argument("--output_dir", type=Path, help="Output directory")
    parser.add_argument("--prompt", type=str, default="a photo of", help="Generation prompt")

    args = parser.parse_args()

    captioner = BLIP2Captioner(
        trigger_word=args.trigger_word
    )

    captions = captioner.batch_caption(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        save_individual_files=True,
        save_combined_file=True
    )

    print(f"\n✓ Captioned {len(captions)} images")


if __name__ == "__main__":
    main()
