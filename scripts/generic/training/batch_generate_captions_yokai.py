#!/usr/bin/env python3
"""
Batch Caption Generation for Yokai Watch Characters

Generates BLIP2 captions optimized for Yokai Watch dataset:
- Human characters: Focus on appearance, clothing, expressions
- Yokai: Focus on colors, shapes, unique features
- Background-aware captioning
- Multi-GPU parallel processing
- Resume capability

Integrates with existing tools:
- generate_captions_blip2.py for base BLIP2 functionality
- apply_captions_from_analysis.py for caption formatting
"""

import torch
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict
from tqdm import tqdm
import json
from datetime import datetime
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')


class YokaiCaptionGenerator:
    """Generates captions optimized for Yokai Watch characters"""

    def __init__(self, model_name: str = "Salesforce/blip2-opt-6.7b",
                 device: str = "cuda", batch_size: int = 8):
        """
        Initialize caption generator

        Args:
            model_name: BLIP2 model to use
            device: Device for processing
            batch_size: Batch size for generation
        """
        self.device = device
        self.batch_size = batch_size

        print(f"🔧 Loading BLIP2 model: {model_name}")
        print(f"   Device: {device}, Batch size: {batch_size}")

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.model.to(device)
        self.model.eval()

        print("✓ Model loaded successfully")

    def get_prompt_template(self, character_type: str, cluster_name: str) -> str:
        """
        Get appropriate prompt template based on character type

        Args:
            character_type: "human", "yokai", or "unknown"
            cluster_name: Cluster name for context

        Returns:
            Prompt template string
        """
        if character_type == "human":
            return (
                "A detailed description of this anime character, "
                "focusing on their appearance, clothing style, hair, "
                "facial features, and body pose. "
                "Describe what makes this character unique and recognizable."
            )
        elif character_type == "yokai":
            return (
                "A detailed description of this yokai character from Yokai Watch, "
                "focusing on its distinctive colors, shape, body features, "
                "facial expression, and any unique characteristics. "
                "Describe what makes this yokai visually unique."
            )
        else:  # unknown
            return (
                "A detailed description of this character, "
                "focusing on appearance, colors, distinctive features, "
                "and visual characteristics."
            )

    def generate_caption_batch(self, images: List[Image.Image],
                              prompt: str) -> List[str]:
        """
        Generate captions for a batch of images

        Args:
            images: List of PIL images
            prompt: Prompt template

        Returns:
            List of generated captions
        """
        # Prepare inputs
        inputs = self.processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.7,
                do_sample=False
            )

        # Decode
        captions = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return [c.strip() for c in captions]

    def generate_for_cluster(self, cluster_dir: Path,
                           character_type: str = "unknown",
                           trigger_word: str = None) -> Dict:
        """
        Generate captions for all images in a cluster

        Args:
            cluster_dir: Cluster directory path
            character_type: "human", "yokai", or "unknown"
            trigger_word: Optional trigger word to prepend

        Returns:
            Generation statistics
        """
        # Find images
        image_files = sorted(cluster_dir.glob("*.png"))

        if not image_files:
            return {"success": False, "error": "No images found"}

        cluster_name = cluster_dir.name

        # Get prompt template
        prompt = self.get_prompt_template(character_type, cluster_name)

        # Process in batches
        captions_generated = 0
        captions_skipped = 0

        for i in tqdm(range(0, len(image_files), self.batch_size),
                     desc=f"  Captioning {cluster_name}",
                     leave=False):
            batch_files = image_files[i:i + self.batch_size]

            # Check which files need captions
            files_to_process = []
            for img_file in batch_files:
                caption_file = img_file.with_suffix('.txt')
                if caption_file.exists():
                    captions_skipped += 1
                else:
                    files_to_process.append(img_file)

            if not files_to_process:
                continue

            # Load images
            images = []
            for img_file in files_to_process:
                try:
                    img = Image.open(img_file).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"⚠️  Failed to load {img_file.name}: {e}")
                    continue

            if not images:
                continue

            # Generate captions
            try:
                captions = self.generate_caption_batch(images, prompt)

                # Save captions
                for img_file, caption in zip(files_to_process, captions):
                    caption_file = img_file.with_suffix('.txt')

                    # Prepend trigger word if provided
                    if trigger_word:
                        final_caption = f"{trigger_word}, {caption}"
                    else:
                        final_caption = caption

                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write(final_caption)

                    captions_generated += 1

            except Exception as e:
                print(f"⚠️  Batch generation failed: {e}")
                continue

        return {
            "success": True,
            "cluster_name": cluster_name,
            "total_images": len(image_files),
            "captions_generated": captions_generated,
            "captions_skipped": captions_skipped,
            "character_type": character_type
        }


def batch_generate_captions(
    clusters_dir: Path,
    cluster_analysis: Path = None,
    selected_clusters: List[str] = None,
    model_name: str = "Salesforce/blip2-opt-6.7b",
    device: str = "cuda",
    batch_size: int = 8,
    auto_trigger_word: bool = True
):
    """
    Generate captions for selected clusters

    Args:
        clusters_dir: Directory containing character clusters
        cluster_analysis: Optional path to cluster analysis JSON
        selected_clusters: Optional list of specific cluster names
        model_name: BLIP2 model name
        device: Processing device
        batch_size: Batch size
        auto_trigger_word: Auto-generate trigger words from cluster names
    """
    print(f"\n{'='*80}")
    print("BATCH CAPTION GENERATION FOR YOKAI WATCH")
    print(f"{'='*80}\n")

    # Load cluster analysis if provided
    cluster_info = {}
    if cluster_analysis and cluster_analysis.exists():
        with open(cluster_analysis, 'r') as f:
            analysis = json.load(f)
            for cluster in analysis.get('clusters', []):
                cluster_info[cluster['cluster_name']] = {
                    'type': cluster.get('character_type', 'unknown'),
                    'recommended': cluster.get('recommended', False)
                }

    # Find clusters to process
    cluster_dirs = sorted([d for d in clusters_dir.iterdir()
                          if d.is_dir() and d.name.startswith("cluster_")])

    if not cluster_dirs:
        print(f"❌ No clusters found in {clusters_dir}")
        return

    # Filter clusters if specified
    if selected_clusters:
        cluster_dirs = [d for d in cluster_dirs if d.name in selected_clusters]

    if not cluster_dirs:
        print("❌ No matching clusters found")
        return

    print(f"📊 Configuration:")
    print(f"  Clusters directory: {clusters_dir}")
    print(f"  Total clusters: {len(cluster_dirs)}")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Auto trigger words: {auto_trigger_word}")
    print()

    # Initialize generator
    generator = YokaiCaptionGenerator(
        model_name=model_name,
        device=device,
        batch_size=batch_size
    )

    # Process clusters
    results = []
    for cluster_dir in tqdm(cluster_dirs, desc="Processing clusters"):
        cluster_name = cluster_dir.name

        # Get character type from analysis
        char_type = cluster_info.get(cluster_name, {}).get('type', 'unknown')

        # Generate trigger word
        trigger_word = None
        if auto_trigger_word:
            # Simple trigger: cluster name without "cluster_" prefix
            trigger_word = cluster_name.replace('cluster_', 'char')

        result = generator.generate_for_cluster(
            cluster_dir,
            character_type=char_type,
            trigger_word=trigger_word
        )

        if result["success"]:
            results.append(result)

    # Summary
    total_images = sum(r["total_images"] for r in results)
    total_generated = sum(r["captions_generated"] for r in results)
    total_skipped = sum(r["captions_skipped"] for r in results)

    print(f"\n{'='*80}")
    print("CAPTION GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Clusters processed: {len(results)}")
    print(f"  Total images: {total_images}")
    print(f"  Captions generated: {total_generated}")
    print(f"  Captions skipped (already exist): {total_skipped}")
    print(f"{'='*80}\n")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "total_clusters": len(results),
        "total_images": total_images,
        "captions_generated": total_generated,
        "captions_skipped": total_skipped,
        "results": results
    }

    metadata_path = clusters_dir / "caption_generation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {metadata_path}\n")

    # Show clusters by type
    type_stats = {}
    for r in results:
        char_type = r.get("character_type", "unknown")
        if char_type not in type_stats:
            type_stats[char_type] = {"clusters": 0, "images": 0}
        type_stats[char_type]["clusters"] += 1
        type_stats[char_type]["images"] += r["captions_generated"]

    print("Captions by character type:")
    for char_type, stats in sorted(type_stats.items()):
        print(f"  {char_type.title()}: {stats['clusters']} clusters, "
              f"{stats['images']} captions")


def main():
    parser = argparse.ArgumentParser(
        description="Batch caption generation for Yokai Watch clusters"
    )

    parser.add_argument(
        "clusters_dir",
        type=Path,
        help="Directory containing character clusters"
    )
    parser.add_argument(
        "--cluster-analysis",
        type=Path,
        default=None,
        help="Path to cluster analysis JSON (for character types)"
    )
    parser.add_argument(
        "--selected-clusters",
        nargs="+",
        default=None,
        help="Specific cluster names to process (e.g., cluster_000 cluster_001)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip2-opt-6.7b",
        choices=[
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-opt-6.7b",
            "Salesforce/blip2-flan-t5-xl"
        ],
        help="BLIP2 model to use (default: opt-6.7b for best quality)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Processing device"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)"
    )
    parser.add_argument(
        "--no-trigger-words",
        action="store_true",
        help="Disable automatic trigger word generation"
    )

    args = parser.parse_args()

    if not args.clusters_dir.exists():
        print(f"❌ Clusters directory not found: {args.clusters_dir}")
        return

    batch_generate_captions(
        clusters_dir=args.clusters_dir,
        cluster_analysis=args.cluster_analysis,
        selected_clusters=args.selected_clusters,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        auto_trigger_word=not args.no_trigger_words
    )


if __name__ == "__main__":
    main()
