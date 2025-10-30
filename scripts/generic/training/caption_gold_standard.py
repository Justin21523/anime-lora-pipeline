"""
Generate captions for gold standard images using WD14 Tagger v3
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.config_loader import load_character_config
from core.utils.logger import setup_logger, print_section, print_success, print_info
from core.utils.path_utils import get_character_paths, ensure_dir, list_images
from core.pipeline.auto_captioner import AutoCaptioner


def caption_gold_standard(character_name: str, style: str = "detailed"):
    """
    Generate captions for gold standard images

    Args:
        character_name: Character name (e.g., 'endou_mamoru')
        style: Caption style ('minimal' or 'detailed')
    """
    # Setup logger
    log_dir = Path("outputs/logs")
    ensure_dir(log_dir)
    setup_logger(log_file=log_dir / "caption_gold_standard.log", level="INFO")

    print_section(f"Captioning Gold Standard: {character_name}")

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

    # Get thresholds from config
    general_threshold = char_config.character_filtering.stage1_tagger.confidence_threshold
    character_threshold = 0.85  # Keep high for character tags

    print_info(f"Trigger word: {trigger_word}")
    print_info(f"Caption style: {style}")
    print_info(f"General threshold: {general_threshold}")
    print_info(f"Tag blacklist: {tag_blacklist}")

    # Count images
    images = list_images(gold_standard_dir, recursive=False)
    print_info(f"Found {len(images)} images in {gold_standard_dir}")

    # Initialize captioner
    captioner = AutoCaptioner(
        trigger_word=trigger_word,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        device=config.hardware.gpu.device
    )

    # Generate captions
    captions = captioner.batch_caption(
        image_dir=gold_standard_dir,
        output_dir=gold_standard_dir,  # Save in same directory
        style=style,
        tag_blacklist=tag_blacklist,
        save_individual_files=True,
        save_combined_file=True
    )

    print_success(f"✓ Generated {len(captions)} captions")
    print_info(f"Caption files saved to: {gold_standard_dir}")

    # Show sample captions
    print_section("Sample Captions")
    for i, (img_path, caption) in enumerate(list(captions.items())[:3]):
        print(f"\n{img_path.name}:")
        # Truncate long captions for display
        if len(caption) > 100:
            print(f"  {caption[:100]}...")
        else:
            print(f"  {caption}")

        if i >= 2:  # Only show 3 samples
            break

    print_info(f"\nTotal: {len(captions)} images captioned")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate captions for gold standard images"
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

    args = parser.parse_args()

    caption_gold_standard(args.character, args.style)


if __name__ == "__main__":
    main()
