#!/usr/bin/env python3
"""
Generate captions using BLIP-2 in the blip2-env environment
Run with: conda run -n blip2-env python3 scripts/tools/generate_captions_blip2.py <character_name>
"""

import sys
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.config_loader import load_character_config
from core.utils.logger import setup_logger, print_section, print_success, print_info, print_error
from core.utils.path_utils import get_character_paths, list_images


def generate_captions_blip2(
    character_name: str,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    prompt: str = "a photo of",
    max_length: int = 50
):
    """
    Generate BLIP-2 captions for character images

    Args:
        character_name: Character name (e.g., 'endou_mamoru')
        model_name: BLIP-2 model to use
        prompt: Generation prompt
        max_length: Maximum caption length
    """
    # Setup
    setup_logger(Path("outputs/logs") / "blip2_captions.log", level="INFO")

    print_section(f"BLIP-2 Caption Generation: {character_name}")

    # Load config
    config = load_character_config(character_name)
    trigger_word = config.character_config.character.trigger_word

    # Get paths
    char_paths = get_character_paths(character_name)
    gold_standard_version = config.character_config.data_sources.gold_standard.version
    image_dir = char_paths['gold_standard'] / gold_standard_version / "images"

    if not image_dir.exists():
        print_error(f"Image directory not found: {image_dir}")
        return

    print_info(f"Image directory: {image_dir}")
    print_info(f"Trigger word: {trigger_word}")
    print_info(f"Model: {model_name}")

    # Load BLIP-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_info(f"Device: {device}")

    print_section("Loading BLIP-2 Model")
    print_info("This may take a few minutes for first download...")

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    model.eval()

    print_success("Model loaded successfully")

    # Get images
    images = list_images(image_dir, recursive=False)
    print_info(f"Found {len(images)} images")

    print_section("Generating Captions")

    # Generate captions
    count = 0
    for img_path in images:
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Generate caption
            inputs = processor(image, text=prompt, return_tensors="pt").to(
                device, torch.float16 if device == "cuda" else torch.float32
            )

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5
                )

            caption = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            # Add trigger word
            full_caption = f"{trigger_word}, {caption}"

            # Save caption
            caption_file = img_path.with_suffix('.txt')
            caption_file.write_text(full_caption, encoding='utf-8')

            count += 1

            if count % 10 == 0:
                print_info(f"Processed {count}/{len(images)} images...")

        except Exception as e:
            print_error(f"Error processing {img_path.name}: {e}")

    print_success(f"Generated {count} captions")
    print_info(f"Saved to: {image_dir}")

    # Show samples
    print_section("Sample Captions")
    for img_path in list(images)[:3]:
        caption_file = img_path.with_suffix('.txt')
        if caption_file.exists():
            caption = caption_file.read_text(encoding='utf-8')
            print_info(f"{img_path.name}:")
            print(f"  {caption}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate BLIP-2 Captions")
    parser.add_argument(
        "character",
        type=str,
        help="Character name (e.g., endou_mamoru)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip2-opt-2.7b",
        help="BLIP-2 model (default: blip2-opt-2.7b)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of",
        help="Generation prompt (default: 'a photo of')"
    )

    args = parser.parse_args()

    generate_captions_blip2(
        character_name=args.character,
        model_name=args.model,
        prompt=args.prompt
    )


if __name__ == "__main__":
    main()
