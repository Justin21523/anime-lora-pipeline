"""
Character Filter using two-stage filtering: WD14 Tagger + CLIP similarity
"""

import torch
import clip
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import random

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.logger import get_logger, create_progress_bar, print_success, print_error, print_section, print_info, print_warning
from core.utils.path_utils import ensure_dir, list_images, safe_copy, save_json
from core.pipeline.auto_captioner import WD14Tagger


logger = get_logger("CharacterFilter")


class CLIPMatcher:
    """
    CLIP-based image similarity matcher
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32
    ):
        """
        Initialize CLIP matcher

        Args:
            model_name: CLIP model name
            device: Device for inference
            batch_size: Batch size for encoding
        """
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        logger.success("CLIP model loaded")

    @torch.no_grad()
    def encode_image(self, image_path: Path) -> torch.Tensor:
        """
        Encode single image to CLIP embedding

        Args:
            image_path: Path to image file

        Returns:
            Normalized CLIP embedding
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Encode
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu()

        except Exception as e:
            logger.error(f"Error encoding {image_path.name}: {e}")
            return None

    @torch.no_grad()
    def encode_images_batch(self, image_paths: List[Path]) -> torch.Tensor:
        """
        Encode multiple images in batches

        Args:
            image_paths: List of image paths

        Returns:
            Tensor of normalized embeddings [N, D]
        """
        all_features = []

        with create_progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Encoding images with CLIP...",
                total=len(image_paths)
            )

            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                batch_images = []

                # Load and preprocess batch
                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_input = self.preprocess(img)
                        batch_images.append(img_input)
                    except Exception as e:
                        logger.error(f"Error loading {img_path.name}: {e}")
                        continue

                if not batch_images:
                    continue

                # Encode batch
                batch_tensor = torch.stack(batch_images).to(self.device)
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

                all_features.append(features.cpu())
                progress.update(task, advance=len(batch_paths))

        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.empty(0, 768)  # Empty tensor

    def compute_similarity(
        self,
        query_embedding: torch.Tensor,
        reference_embeddings: torch.Tensor
    ) -> float:
        """
        Compute average cosine similarity

        Args:
            query_embedding: Query image embedding [1, D]
            reference_embeddings: Reference embeddings [N, D]

        Returns:
            Average similarity score
        """
        similarities = (query_embedding @ reference_embeddings.T).squeeze()

        if similarities.dim() == 0:
            return float(similarities)
        else:
            return float(similarities.mean())


class CharacterFilter:
    """
    Two-stage character filtering: WD14 Tagger + CLIP
    """

    def __init__(
        self,
        gold_standard_dir: Path,
        required_tags: List[str],
        forbidden_tags: List[str],
        tagger_threshold: float = 0.6,
        clip_threshold: float = 0.75,
        clip_model: str = "ViT-L/14",
        top_k_references: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize CharacterFilter

        Args:
            gold_standard_dir: Directory with gold standard images
            required_tags: Tags that must be present (Stage 1)
            forbidden_tags: Tags that must not be present (Stage 1)
            tagger_threshold: WD14 confidence threshold (Stage 1)
            clip_threshold: CLIP similarity threshold (Stage 2)
            clip_model: CLIP model name
            top_k_references: Use top K references for CLIP matching
            device: Device for inference
        """
        self.gold_standard_dir = gold_standard_dir
        self.required_tags = required_tags
        self.forbidden_tags = forbidden_tags
        self.tagger_threshold = tagger_threshold
        self.clip_threshold = clip_threshold
        self.top_k_references = top_k_references
        self.device = device

        # Initialize models
        logger.info("Initializing CharacterFilter...")
        self.tagger = WD14Tagger(device=device)
        self.clip_matcher = CLIPMatcher(model_name=clip_model, device=device)

        # Load gold standard references
        self.reference_images = list_images(gold_standard_dir, recursive=True)

        if not self.reference_images:
            raise ValueError(f"No reference images found in {gold_standard_dir}")

        # Sample top K references if too many
        if len(self.reference_images) > top_k_references:
            self.reference_images = random.sample(self.reference_images, top_k_references)

        logger.info(f"Using {len(self.reference_images)} reference images")

        # Encode references
        print_section("Encoding Gold Standard References")
        self.reference_embeddings = self.clip_matcher.encode_images_batch(self.reference_images)
        print_success(f"Encoded {len(self.reference_images)} reference images")

    def stage1_filter_by_tags(
        self,
        image_paths: List[Path]
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """
        Stage 1: Filter by WD14 tags

        Args:
            image_paths: List of images to filter

        Returns:
            Tuple of (passed_images, failed_images_with_reasons)
        """
        print_section("Stage 1: Tag-based Filtering")

        passed = []
        failed = []

        with create_progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Checking tags...",
                total=len(image_paths)
            )

            for img_path in image_paths:
                try:
                    # Get tags
                    tags = self.tagger.predict(img_path, general_threshold=self.tagger_threshold)

                    tag_names = set(tags.keys())

                    # Check required tags
                    missing_required = set(self.required_tags) - tag_names
                    if missing_required:
                        failed.append((img_path, f"missing_tags: {missing_required}"))
                        progress.update(task, advance=1)
                        continue

                    # Check forbidden tags
                    found_forbidden = tag_names & set(self.forbidden_tags)
                    if found_forbidden:
                        failed.append((img_path, f"forbidden_tags: {found_forbidden}"))
                        progress.update(task, advance=1)
                        continue

                    # Passed Stage 1
                    passed.append(img_path)

                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
                    failed.append((img_path, f"error: {str(e)}"))

                progress.update(task, advance=1)

        print_success(f"Stage 1 passed: {len(passed)}/{len(image_paths)}")
        return passed, failed

    def stage2_filter_by_similarity(
        self,
        image_paths: List[Path]
    ) -> Tuple[List[Path], List[Tuple[Path, float]]]:
        """
        Stage 2: Filter by CLIP similarity to gold standard

        Args:
            image_paths: List of images that passed Stage 1

        Returns:
            Tuple of (passed_images, failed_images_with_scores)
        """
        print_section("Stage 2: CLIP Similarity Filtering")

        if not image_paths:
            print_warning("No images to process in Stage 2")
            return [], []

        passed = []
        failed = []

        # Encode query images
        query_embeddings = self.clip_matcher.encode_images_batch(image_paths)

        print_info(f"Computing similarities...")

        with create_progress_bar() as progress:
            task = progress.add_task(
                "[cyan]Matching characters...",
                total=len(image_paths)
            )

            for i, img_path in enumerate(image_paths):
                try:
                    query_emb = query_embeddings[i:i+1]

                    # Compute similarity with references
                    similarity = self.clip_matcher.compute_similarity(
                        query_emb,
                        self.reference_embeddings
                    )

                    if similarity >= self.clip_threshold:
                        passed.append(img_path)
                    else:
                        failed.append((img_path, similarity))

                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
                    failed.append((img_path, 0.0))

                progress.update(task, advance=1)

        print_success(f"Stage 2 passed: {len(passed)}/{len(image_paths)}")
        return passed, failed

    def filter_images(
        self,
        input_dir: Path,
        output_dir: Path,
        copy_files: bool = True,
        save_report: bool = True
    ) -> Dict[str, any]:
        """
        Filter images using two-stage pipeline

        Args:
            input_dir: Input directory with candidate images
            output_dir: Output directory for filtered images
            copy_files: Copy matched images to output
            save_report: Save filtering report

        Returns:
            Filtering statistics
        """
        ensure_dir(output_dir)

        print_section(f"Filtering Images: {input_dir.name}")

        # Get all images
        all_images = list_images(input_dir, recursive=False)
        print_info(f"Found {len(all_images)} candidate images")

        # Stage 1: Tag filtering
        stage1_passed, stage1_failed = self.stage1_filter_by_tags(all_images)

        # Stage 2: CLIP similarity
        stage2_passed, stage2_failed = self.stage2_filter_by_similarity(stage1_passed)

        # Statistics
        stats = {
            'input_images': len(all_images),
            'stage1_passed': len(stage1_passed),
            'stage1_failed': len(stage1_failed),
            'stage2_passed': len(stage2_passed),
            'stage2_failed': len(stage2_failed),
            'final_matched': len(stage2_passed)
        }

        # Copy matched images
        if copy_files and stage2_passed:
            print_section("Copying Matched Images")

            with create_progress_bar() as progress:
                task = progress.add_task(
                    "[cyan]Copying files...",
                    total=len(stage2_passed)
                )

                for img_path in stage2_passed:
                    dst_path = output_dir / img_path.name
                    safe_copy(img_path, dst_path, overwrite=True)
                    progress.update(task, advance=1)

            print_success(f"Copied {len(stage2_passed)} matched images")

        # Save report
        if save_report:
            report = {
                'statistics': stats,
                'matched_images': [str(img) for img in stage2_passed],
                'stage1_failed': [(str(img), reason) for img, reason in stage1_failed],
                'stage2_failed': [(str(img), score) for img, score in stage2_failed]
            }

            report_path = output_dir / "filtering_report.json"
            save_json(report, report_path)
            print_info(f"Report saved to: {report_path}")

        # Summary
        print_section("Filtering Summary")
        print_info(f"Input images: {stats['input_images']}")
        print_info(f"Stage 1 passed: {stats['stage1_passed']} ({stats['stage1_passed']/stats['input_images']*100:.1f}%)")
        print_info(f"Stage 2 passed: {stats['stage2_passed']} ({stats['stage2_passed']/stats['stage1_passed']*100:.1f}% of stage 1)")
        print_success(f"Final matched: {stats['final_matched']} ({stats['final_matched']/stats['input_images']*100:.1f}% of total)")

        return stats


def main():
    """Test CharacterFilter"""
    import argparse

    parser = argparse.ArgumentParser(description="Filter character images")
    parser.add_argument("input_dir", type=Path, help="Input directory")
    parser.add_argument("--gold-standard", type=Path, required=True, help="Gold standard directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--required-tags", nargs="+", default=["1boy", "male_focus"], help="Required tags")
    parser.add_argument("--forbidden-tags", nargs="+", default=["multiple_boys"], help="Forbidden tags")
    parser.add_argument("--clip-threshold", type=float, default=0.75, help="CLIP similarity threshold")

    args = parser.parse_args()

    filter_pipeline = CharacterFilter(
        gold_standard_dir=args.gold_standard,
        required_tags=args.required_tags,
        forbidden_tags=args.forbidden_tags,
        clip_threshold=args.clip_threshold
    )

    stats = filter_pipeline.filter_images(
        input_dir=args.input_dir,
        output_dir=args.output,
        copy_files=True,
        save_report=True
    )

    print(f"\n✓ Matched {stats['final_matched']} images")


if __name__ == "__main__":
    main()
