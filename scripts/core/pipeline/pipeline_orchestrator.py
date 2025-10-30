"""
Pipeline Orchestrator - Coordinates all modules for end-to-end processing
"""

import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.config_loader import load_character_config
from core.utils.logger import setup_logger, get_logger, print_section, print_success, print_error, print_info
from core.utils.path_utils import ensure_dir, get_character_paths, save_json

from core.pipeline.video_processor import VideoProcessor
from core.pipeline.image_cleaner import ImageCleaner
from core.pipeline.character_filter import CharacterFilter
from core.pipeline.auto_captioner import AutoCaptioner


logger = get_logger("PipelineOrchestrator")


class PipelineOrchestrator:
    """
    Orchestrates the complete pipeline from video to training-ready dataset
    """

    def __init__(self, character_name: str):
        """
        Initialize pipeline for a character

        Args:
            character_name: Character name (e.g., 'endou_mamoru')
        """
        self.character_name = character_name

        # Load configuration
        logger.info(f"Loading configuration for {character_name}")
        self.config = load_character_config(character_name)
        self.char_config = self.config.character_config

        # Get paths
        self.char_paths = get_character_paths(character_name)

        # Pipeline statistics
        self.stats = {
            'character': character_name,
            'start_time': None,
            'end_time': None,
            'stages': {}
        }

    def run_stage(self, stage_name: str, stage_func):
        """
        Run a pipeline stage with error handling

        Args:
            stage_name: Name of the stage
            stage_func: Function to execute

        Returns:
            Stage result
        """
        print_section(f"Stage: {stage_name}")
        logger.info(f"Starting stage: {stage_name}")

        start_time = time.time()

        try:
            result = stage_func()

            elapsed_time = time.time() - start_time

            self.stats['stages'][stage_name] = {
                'status': 'success',
                'elapsed_time': elapsed_time,
                'result': result
            }

            print_success(f"Completed {stage_name} in {elapsed_time:.2f}s")
            return result

        except Exception as e:
            elapsed_time = time.time() - start_time

            logger.error(f"Error in stage {stage_name}: {e}")

            self.stats['stages'][stage_name] = {
                'status': 'failed',
                'elapsed_time': elapsed_time,
                'error': str(e)
            }

            print_error(f"Failed {stage_name}: {e}")
            raise

    def stage_video_extraction(self, video_paths: List[Path]) -> Path:
        """
        Stage 1: Extract frames from videos

        Args:
            video_paths: List of video file paths

        Returns:
            Directory with extracted frames
        """
        # Get config
        video_config = self.char_config.video_processing

        # Create processor
        processor = VideoProcessor(
            min_scene_length=video_config.frame_extraction.min_scene_length,
            threshold=27.0,
            min_resolution=video_config.quality_filters.min_resolution,
            blur_threshold=video_config.quality_filters.blur_threshold,
            brightness_range=video_config.quality_filters.brightness_range
        )

        # Output directory
        output_dir = self.char_paths['auto_collected'] / "01_extracted_frames"
        ensure_dir(output_dir)

        # Process videos
        all_stats = processor.batch_process_videos(
            video_paths=video_paths,
            output_root=output_dir,
            create_subdirs=True
        )

        logger.info(f"Extracted frames to {output_dir}")
        return output_dir

    def stage_image_cleaning(self, input_dir: Path) -> Path:
        """
        Stage 2: Clean and deduplicate images

        Args:
            input_dir: Directory with raw extracted frames

        Returns:
            Directory with cleaned images
        """
        # Create cleaner
        cleaner = ImageCleaner(
            min_resolution=self.char_config.video_processing.quality_filters.min_resolution,
            blur_threshold=self.char_config.video_processing.quality_filters.blur_threshold,
            brightness_range=self.char_config.video_processing.quality_filters.brightness_range
        )

        # Output directory
        output_dir = self.char_paths['auto_collected'] / "02_cleaned"
        ensure_dir(output_dir)

        # Clean images
        stats = cleaner.clean_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            remove_duplicates=True,
            remove_low_quality=True,
            copy_files=True,
            save_report=True
        )

        logger.info(f"Cleaned {stats['output_images']} images to {output_dir}")
        return output_dir

    def stage_character_filtering(self, input_dir: Path) -> Path:
        """
        Stage 3: Filter for target character

        Args:
            input_dir: Directory with cleaned images

        Returns:
            Directory with character-matched images
        """
        # Get gold standard path
        gold_standard_version = self.char_config.data_sources.gold_standard.version
        gold_standard_dir = self.char_paths['gold_standard'] / gold_standard_version / "images"

        if not gold_standard_dir.exists():
            raise FileNotFoundError(f"Gold standard not found: {gold_standard_dir}")

        # Get filtering config
        filter_config = self.char_config.character_filtering

        # Create filter
        char_filter = CharacterFilter(
            gold_standard_dir=gold_standard_dir,
            required_tags=self.char_config.character.filter_tags.required,
            forbidden_tags=self.char_config.character.filter_tags.forbidden,
            tagger_threshold=filter_config.stage1_tagger.confidence_threshold,
            clip_threshold=filter_config.stage2_clip.similarity_threshold,
            clip_model=filter_config.stage2_clip.model,
            top_k_references=filter_config.stage2_clip.top_k_references,
            device=self.config.hardware.gpu.device
        )

        # Output directory
        output_dir = self.char_paths['auto_collected'] / "03_character_matched"
        ensure_dir(output_dir)

        # Filter images
        stats = char_filter.filter_images(
            input_dir=input_dir,
            output_dir=output_dir,
            copy_files=True,
            save_report=True
        )

        logger.info(f"Matched {stats['final_matched']} character images to {output_dir}")
        return output_dir

    def stage_auto_captioning(self, input_dir: Path) -> Path:
        """
        Stage 4: Generate captions for images

        Args:
            input_dir: Directory with character-matched images

        Returns:
            Directory with captioned images
        """
        # Get caption config
        caption_config = self.char_config.auto_caption

        # Create captioner
        captioner = AutoCaptioner(
            trigger_word=self.char_config.character.trigger_word,
            general_threshold=self.char_config.character_filtering.stage1_tagger.confidence_threshold,
            device=self.config.hardware.gpu.device
        )

        # Caption images (save in same directory)
        captions = captioner.batch_caption(
            image_dir=input_dir,
            output_dir=input_dir,  # Save captions in same directory
            style=caption_config.style,
            tag_blacklist=caption_config.get('tag_blacklist', []),
            save_individual_files=True,
            save_combined_file=True
        )

        logger.info(f"Generated {len(captions)} captions")
        return input_dir

    def stage_prepare_training_data(self, input_dir: Path) -> Path:
        """
        Stage 5: Organize data for training

        Args:
            input_dir: Directory with captioned images

        Returns:
            Training-ready directory
        """
        from shutil import copy2

        # Output directory (kohya_ss format: <repeat>_<class_name>)
        repeat_count = 10  # How many times to repeat images per epoch
        class_name = self.char_config.character.trigger_word

        output_dir = self.char_paths['training_ready'] / f"{repeat_count}_{class_name}"
        ensure_dir(output_dir)

        # Copy all images and captions
        from core.utils.path_utils import list_images

        images = list_images(input_dir, recursive=False)

        print_info(f"Preparing {len(images)} images for training")

        for img_path in images:
            # Copy image
            dst_img = output_dir / img_path.name
            copy2(img_path, dst_img)

            # Copy caption if exists
            caption_file = img_path.with_suffix('.txt')
            if caption_file.exists():
                dst_caption = output_dir / caption_file.name
                copy2(caption_file, dst_caption)

        logger.info(f"Training data ready at {output_dir}")
        return output_dir

    def run_full_pipeline(
        self,
        video_paths: Optional[List[Path]] = None,
        skip_video_extraction: bool = False
    ) -> Dict:
        """
        Run complete pipeline

        Args:
            video_paths: List of video files to process (if None, use config)
            skip_video_extraction: Skip video processing (use existing frames)

        Returns:
            Pipeline statistics
        """
        self.stats['start_time'] = datetime.now().isoformat()

        print_section(f"Pipeline: {self.character_name}")
        print_info(f"Character: {self.char_config.character.name}")
        print_info(f"Trigger word: {self.char_config.character.trigger_word}")

        try:
            # Get video paths from config if not provided
            if video_paths is None and not skip_video_extraction:
                video_paths = [
                    Path(v.path) for v in self.char_config.data_sources.videos
                ]

            # Stage 1: Video extraction
            if not skip_video_extraction:
                frames_dir = self.run_stage(
                    "Video Frame Extraction",
                    lambda: self.stage_video_extraction(video_paths)
                )
            else:
                frames_dir = self.char_paths['auto_collected'] / "01_extracted_frames"
                print_info(f"Skipping video extraction, using: {frames_dir}")

            # Stage 2: Image cleaning
            cleaned_dir = self.run_stage(
                "Image Cleaning",
                lambda: self.stage_image_cleaning(frames_dir)
            )

            # Stage 3: Character filtering
            matched_dir = self.run_stage(
                "Character Filtering",
                lambda: self.stage_character_filtering(cleaned_dir)
            )

            # Stage 4: Auto captioning
            captioned_dir = self.run_stage(
                "Auto Captioning",
                lambda: self.stage_auto_captioning(matched_dir)
            )

            # Stage 5: Prepare training data
            training_dir = self.run_stage(
                "Prepare Training Data",
                lambda: self.stage_prepare_training_data(captioned_dir)
            )

            self.stats['status'] = 'success'
            self.stats['training_ready_dir'] = str(training_dir)

        except Exception as e:
            self.stats['status'] = 'failed'
            self.stats['error'] = str(e)
            logger.error(f"Pipeline failed: {e}")

        finally:
            self.stats['end_time'] = datetime.now().isoformat()

            # Save statistics
            stats_file = self.char_paths['root'] / "pipeline_stats.json"
            save_json(self.stats, stats_file)
            print_info(f"Statistics saved to: {stats_file}")

        # Print summary
        print_section("Pipeline Summary")
        for stage_name, stage_info in self.stats['stages'].items():
            status = stage_info['status']
            elapsed = stage_info.get('elapsed_time', 0)

            if status == 'success':
                print_success(f"{stage_name}: {elapsed:.2f}s")
            else:
                print_error(f"{stage_name}: FAILED")

        return self.stats


def main():
    """Test PipelineOrchestrator"""
    import argparse

    parser = argparse.ArgumentParser(description="Run complete data processing pipeline")
    parser.add_argument("character", type=str, help="Character name (e.g., endou_mamoru)")
    parser.add_argument("--videos", nargs="+", type=Path, help="Video files to process")
    parser.add_argument("--skip-video", action="store_true", help="Skip video extraction")

    args = parser.parse_args()

    # Setup logging
    log_dir = Path("outputs/logs")
    ensure_dir(log_dir)
    setup_logger(log_file=log_dir / f"pipeline_{args.character}.log", level="INFO")

    # Run pipeline
    orchestrator = PipelineOrchestrator(args.character)

    stats = orchestrator.run_full_pipeline(
        video_paths=args.videos,
        skip_video_extraction=args.skip_video
    )

    if stats['status'] == 'success':
        print(f"\n✓ Pipeline completed successfully")
        print(f"  Training data: {stats['training_ready_dir']}")
    else:
        print(f"\n✗ Pipeline failed: {stats.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
