"""
Prepare gold standard images for LoRA training
Generates captions and organizes in kohya_ss format
"""

import sys
import shutil
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.config_loader import load_character_config
from core.utils.logger import setup_logger, print_section, print_success, print_info
from core.utils.path_utils import get_character_paths, ensure_dir, list_images
from core.pipeline.auto_captioner import AutoCaptioner


def prepare_training_data(
    character_name: str,
    caption_style: str = "detailed",
    repeat_count: int = 10
):
    """
    Prepare gold standard images for training

    Args:
        character_name: Character name (e.g., 'endou_mamoru')
        caption_style: Caption style ('minimal' or 'detailed')
        repeat_count: How many times to repeat images per epoch
    """
    # Setup logger
    log_dir = Path("outputs/logs")
    ensure_dir(log_dir)
    setup_logger(log_file=log_dir / "prepare_training_data.log", level="INFO")

    print_section(f"Preparing Training Data: {character_name}")

    # Load config
    config = load_character_config(character_name)
    char_config = config.character_config

    # Get paths
    char_paths = get_character_paths(character_name)
    gold_standard_version = char_config.data_sources.gold_standard.version
    gold_standard_dir = char_paths['gold_standard'] / gold_standard_version / "images"

    if not gold_standard_dir.exists():
        print(f"Error: Gold standard directory not found: {gold_standard_dir}")
        return

    # Get configuration
    trigger_word = char_config.character.trigger_word
    tag_blacklist = char_config.auto_caption.get('tag_blacklist', [])
    general_threshold = char_config.character_filtering.stage1_tagger.confidence_threshold

    print_info(f"Source: {gold_standard_dir}")
    print_info(f"Trigger word: {trigger_word}")
    print_info(f"Caption style: {caption_style}")
    print_info(f"Repeat count: {repeat_count}")

    # Count images
    images = list_images(gold_standard_dir, recursive=False)
    print_info(f"Found {len(images)} images")

    # Step 1: Generate captions if they don't exist
    print_section("Step 1: Generating Captions")

    captioner = AutoCaptioner(
        trigger_word=trigger_word,
        general_threshold=general_threshold,
        device=config.hardware.gpu.device
    )

    captions = captioner.batch_caption(
        image_dir=gold_standard_dir,
        output_dir=gold_standard_dir,  # Save in same directory
        style=caption_style,
        tag_blacklist=tag_blacklist,
        save_individual_files=True,
        save_combined_file=True
    )

    print_success(f"Generated {len(captions)} captions")

    # Step 2: Organize for kohya_ss training
    print_section("Step 2: Organizing Training Data")

    # Output directory format: <repeat>_<class_name>
    # This tells kohya_ss to repeat each image 'repeat_count' times per epoch
    class_name = trigger_word
    training_dir = char_paths['training_ready'] / f"{repeat_count}_{class_name}"

    # Clean and recreate output directory
    if training_dir.exists():
        print_info(f"Removing existing training directory: {training_dir}")
        shutil.rmtree(training_dir)

    ensure_dir(training_dir)

    # Copy images and captions
    copied_count = 0
    for img_path in images:
        # Copy image
        dst_img = training_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        # Copy caption if exists
        caption_file = img_path.with_suffix('.txt')
        if caption_file.exists():
            dst_caption = training_dir / caption_file.name
            shutil.copy2(caption_file, dst_caption)
        else:
            print(f"Warning: No caption for {img_path.name}")

        copied_count += 1

    print_success(f"Copied {copied_count} images to training directory")

    # Summary
    print_section("Training Data Ready")
    print_info(f"Location: {training_dir}")
    print_info(f"Images: {copied_count}")
    print_info(f"Repeats per epoch: {repeat_count}")
    print_info(f"Effective dataset size: {copied_count * repeat_count} per epoch")

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print(f"1. Review captions in: {gold_standard_dir}")
    print(f"2. Training data ready at: {training_dir}")
    print(f"3. Configure LoRA training parameters")
    print(f"4. Run training with kohya_ss")
    print("="*60 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare gold standard images for LoRA training"
    )
    parser.add_argument(
        "character",
        type=str,
        help="Character name (e.g., endou_mamoru)"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="detailed",
        choices=["minimal", "detailed"],
        help="Caption style (default: detailed)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Repeat count per epoch (default: 10)"
    )

    args = parser.parse_args()

    prepare_training_data(
        character_name=args.character,
        caption_style=args.style,
        repeat_count=args.repeat
    )


if __name__ == "__main__":
    main()
