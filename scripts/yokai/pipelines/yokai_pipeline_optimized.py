#!/usr/bin/env python3
"""
Yokai Watch Optimized Pipeline
Directly uses ALL raw frames for character detection, LoRA training, and ControlNet
Only does segmentation on a small random sample for background/character separation
"""

import subprocess
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import time
import shutil
import random


class OptimizedYokaiPipeline:
    """
    Optimized pipeline that:
    1. Uses ALL raw frames directly for character clustering
    2. Does segmentation on small random sample only
    3. Trains LoRA on ALL detected character faces
    4. Trains ControlNet/backgrounds on ALL frames
    """

    def __init__(
        self,
        input_frames_dir: Path,
        base_output_dir: Path,
        segmentation_sample_size: int = 10,
        min_cluster_size: int = 50,
        device: str = "cuda",
        base_model: str = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors"
    ):
        """
        Initialize optimized pipeline

        Args:
            input_frames_dir: Directory with ALL raw frames
            base_output_dir: Base output directory
            segmentation_sample_size: Number of episodes to sample for segmentation
            min_cluster_size: Minimum cluster size for characters
            device: Device to use
            base_model: Base SD model
        """
        self.input_frames_dir = Path(input_frames_dir)
        self.base_output_dir = Path(base_output_dir)
        self.segmentation_sample_size = segmentation_sample_size
        self.min_cluster_size = min_cluster_size
        self.device = device
        self.base_model = base_model

        # Setup paths
        self.warehouse_cache = Path("/mnt/c/AI_LLM_projects/ai_warehouse/cache/yokai-watch")
        self.warehouse_training = Path("/mnt/c/AI_LLM_projects/ai_warehouse/training_data/yokai-watch")
        self.warehouse_outputs = Path("/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch")
        self.warehouse_models = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/yokai-watch")

        # Segmentation output (small sample only)
        self.layered_sample_dir = self.warehouse_cache / "layered_frames_sample"

        # Character faces from ALL frames
        self.character_faces_dir = self.warehouse_training / "character_faces_all"
        self.character_clusters_dir = self.warehouse_training / "character_clusters"

        # Script paths
        self.project_root = Path("/mnt/c/AI_LLM_projects/inazuma-eleven-lora")
        self.segmentation_script = self.project_root / "scripts/tools/layered_segmentation.py"
        self.face_clustering_script = self.project_root / "scripts/tools/anime_face_clustering.py"
        self.ai_analysis_script = self.project_root / "scripts/evaluation/comprehensive_anime_analysis.py"
        self.caption_apply_script = self.project_root / "scripts/tools/apply_captions_from_analysis.py"

        # Conda environment
        self.conda_env = "blip2-env"

        # Create directories
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.warehouse_cache.mkdir(parents=True, exist_ok=True)
        self.warehouse_training.mkdir(parents=True, exist_ok=True)
        self.warehouse_outputs.mkdir(parents=True, exist_ok=True)
        self.warehouse_models.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_file = self.base_output_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        self.log(f"{'='*80}")
        self.log(f"Yokai Watch Optimized Pipeline")
        self.log(f"{'='*80}")
        self.log(f"Input Frames: {self.input_frames_dir}")
        self.log(f"Output Base: {self.base_output_dir}")
        self.log(f"Segmentation Sample: {self.segmentation_sample_size} episodes")
        self.log(f"Min Cluster Size: {self.min_cluster_size}")
        self.log(f"Device: {self.device}")
        self.log(f"Strategy: Use ALL frames for character detection & training")
        self.log(f"{'='*80}\n")

    def log(self, message: str):
        """Log to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run command with logging"""
        self.log(f"\n{'='*80}")
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(str(c) for c in cmd)}")
        self.log(f"{'='*80}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout:
                self.log("STDOUT:")
                self.log(result.stdout)
            if result.stderr:
                self.log("STDERR:")
                self.log(result.stderr)

            elapsed = time.time() - start_time
            self.log(f"✓ {description} completed in {elapsed:.1f}s")
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            self.log(f"✗ {description} failed after {elapsed:.1f}s")
            self.log(f"Return code: {e.returncode}")
            if e.stdout:
                self.log("STDOUT:")
                self.log(e.stdout)
            if e.stderr:
                self.log("STDERR:")
                self.log(e.stderr)
            return False

    def get_all_episodes(self) -> List[Path]:
        """Get all episode folders"""
        episodes = sorted([d for d in self.input_frames_dir.iterdir() if d.is_dir()])
        self.log(f"📺 Found {len(episodes)} total episodes")
        return episodes

    def sample_episodes_for_segmentation(self, all_episodes: List[Path]) -> List[Path]:
        """Randomly sample episodes for segmentation"""
        if self.segmentation_sample_size >= len(all_episodes):
            return all_episodes

        sampled = random.sample(all_episodes, self.segmentation_sample_size)
        sampled.sort()

        self.log(f"\n🎲 Randomly sampled {len(sampled)} episodes for segmentation:")
        for ep in sampled:
            self.log(f"  - {ep.name}")

        return sampled

    def stage1_segmentation_sample(self, sampled_episodes: List[Path]) -> bool:
        """Stage 1: Segment only sampled episodes"""
        self.log(f"\n🎨 Stage 1: Layered Segmentation (Sample Only)")
        self.log(f"Processing {len(sampled_episodes)} episodes to save space")

        self.layered_sample_dir.mkdir(parents=True, exist_ok=True)

        for ep_folder in sampled_episodes:
            self.log(f"  Segmenting: {ep_folder.name}")

            cmd = [
                "conda", "run", "-n", self.conda_env,
                "python3", str(self.segmentation_script),
                str(ep_folder),
                "--output-dir", str(self.layered_sample_dir),
                "--device", self.device
            ]

            success = self.run_command(cmd, f"Segment {ep_folder.name}")
            if not success:
                self.log(f"⚠️  Segmentation failed for {ep_folder.name}")

        return True

    def stage2_character_clustering_all_frames(self) -> bool:
        """Stage 2: Character clustering using ALL raw frames"""
        self.log(f"\n👥 Stage 2: Character Clustering (ALL Frames)")
        self.log(f"Processing ALL frames from: {self.input_frames_dir}")

        # Use anime face clustering on raw frames directly
        self.character_clusters_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.face_clustering_script),
            str(self.input_frames_dir),
            "--output-dir", str(self.character_clusters_dir),
            "--min-cluster-size", str(self.min_cluster_size),
            "--device", self.device,
            "--recursive"  # Process all subdirectories
        ]

        return self.run_command(cmd, "Character Clustering on ALL Frames")

    def get_character_clusters(self) -> List[Path]:
        """Get list of character clusters"""
        if not self.character_clusters_dir.exists():
            return []

        clusters = [d for d in self.character_clusters_dir.iterdir()
                   if d.is_dir() and d.name.startswith("cluster_")]
        clusters.sort()

        self.log(f"\n📊 Found {len(clusters)} character cluster(s):")
        for cluster in clusters:
            num_images = len(list(cluster.glob("*.png"))) + len(list(cluster.glob("*.jpg")))
            self.log(f"  - {cluster.name}: {num_images} images")

        return clusters

    def stage3_ai_analysis(self, cluster_dir: Path) -> bool:
        """Stage 3: AI analysis for character"""
        self.log(f"\n🤖 Stage 3: AI Analysis for {cluster_dir.name}")

        analysis_output = self.warehouse_outputs / cluster_dir.name / "ai_analysis"
        analysis_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.ai_analysis_script),
            str(cluster_dir),
            "--output-dir", str(analysis_output),
            "--device", self.device
        ]

        return self.run_command(cmd, f"AI Analysis for {cluster_dir.name}")

    def stage4_apply_captions(self, cluster_dir: Path) -> bool:
        """Stage 4: Apply captions"""
        self.log(f"\n📝 Stage 4: Apply Captions for {cluster_dir.name}")

        analysis_output = self.warehouse_outputs / cluster_dir.name / "ai_analysis"
        analysis_json = analysis_output / "character_analysis.json"

        if not analysis_json.exists():
            self.log(f"⚠️  Analysis JSON not found: {analysis_json}")
            return False

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.caption_apply_script),
            str(analysis_json),
            str(cluster_dir),
            "--output-dir", str(cluster_dir)
        ]

        return self.run_command(cmd, f"Apply Captions for {cluster_dir.name}")

    def stage5_train_lora(self, cluster_dir: Path) -> bool:
        """Stage 5: Train LoRA on ALL character images"""
        character_name = cluster_dir.name
        self.log(f"\n🎓 Stage 5: LoRA Training for {character_name}")

        num_images = len(list(cluster_dir.glob("*.png"))) + len(list(cluster_dir.glob("*.jpg")))
        num_captions = len(list(cluster_dir.glob("*.txt")))

        self.log(f"Training data: {num_images} images, {num_captions} captions")

        if num_images < self.min_cluster_size or num_captions == 0:
            self.log(f"⚠️  Insufficient training data for {character_name}")
            return False

        model_output_dir = self.warehouse_models / cluster_dir.name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate training steps based on number of images
        # More images = more steps for better quality
        base_steps = 2000
        steps = min(base_steps + (num_images // 10) * 100, 5000)

        self.log(f"Training steps: {steps} (based on {num_images} images)")

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", "-m", "accelerate.commands.launch",
            "--num_cpu_threads_per_process", "1",
            "train_network.py",
            "--pretrained_model_name_or_path", self.base_model,
            "--train_data_dir", str(cluster_dir),
            "--output_dir", str(model_output_dir),
            "--output_name", character_name,
            "--save_model_as", "safetensors",
            "--prior_loss_weight", "1.0",
            "--max_train_steps", str(steps),
            "--learning_rate", "1e-4",
            "--optimizer_type", "AdamW8bit",
            "--xformers",
            "--mixed_precision", "fp16",
            "--cache_latents",
            "--gradient_checkpointing",
            "--save_every_n_epochs", "1",
            "--network_module", "networks.lora",
            "--network_dim", "32",
            "--network_alpha", "16",
            "--train_batch_size", "2",
            "--resolution", "512,512",
            "--enable_bucket",
            "--min_bucket_reso", "256",
            "--max_bucket_reso", "1024",
            "--bucket_reso_steps", "64"
        ]

        return self.run_command(cmd, f"LoRA Training for {character_name}")

    def process_character(self, cluster_dir: Path) -> Dict:
        """Process a single character"""
        character_name = cluster_dir.name
        self.log(f"\n{'#'*80}")
        self.log(f"Processing Character: {character_name}")
        self.log(f"{'#'*80}")

        start_time = time.time()
        results = {
            "character": character_name,
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }

        # AI Analysis
        success = self.stage3_ai_analysis(cluster_dir)
        results["stages"]["ai_analysis"] = success
        if not success:
            results["status"] = "failed_analysis"
            return results

        # Apply Captions
        success = self.stage4_apply_captions(cluster_dir)
        results["stages"]["apply_captions"] = success
        if not success:
            results["status"] = "failed_captions"
            return results

        # Train LoRA
        success = self.stage5_train_lora(cluster_dir)
        results["stages"]["train_lora"] = success
        if not success:
            results["status"] = "failed_training"
            return results

        elapsed = time.time() - start_time
        results["end_time"] = datetime.now().isoformat()
        results["elapsed_seconds"] = elapsed
        results["status"] = "completed"

        self.log(f"\n✅ Character {character_name} complete in {elapsed/60:.1f} minutes")
        return results

    def run(self, max_characters: Optional[int] = None):
        """Run the optimized pipeline"""
        pipeline_start = time.time()

        # Get all episodes
        all_episodes = self.get_all_episodes()

        # Stage 1: Segment ONLY a small random sample
        sampled_episodes = self.sample_episodes_for_segmentation(all_episodes)
        self.log(f"\n💡 Strategy: Segment {len(sampled_episodes)} episodes, ")
        self.log(f"   but use ALL {len(all_episodes)} episodes for character detection")

        if not self.stage1_segmentation_sample(sampled_episodes):
            self.log("⚠️  Segmentation had issues, but continuing with character clustering")

        # Stage 2: Character clustering on ALL frames
        if not self.stage2_character_clustering_all_frames():
            self.log("❌ Character clustering failed")
            return

        # Get character clusters
        clusters = self.get_character_clusters()
        if not clusters:
            self.log("❌ No character clusters found")
            return

        if max_characters:
            clusters = clusters[:max_characters]
            self.log(f"\n📊 Processing first {max_characters} character(s)")

        # Process each character
        all_results = []
        for idx, cluster_dir in enumerate(clusters, 1):
            self.log(f"\n{'='*80}")
            self.log(f"Character {idx}/{len(clusters)}")
            self.log(f"{'='*80}")

            result = self.process_character(cluster_dir)
            all_results.append(result)

            # Save intermediate results
            results_file = self.base_output_dir / "pipeline_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({
                    "pipeline_start": datetime.fromtimestamp(pipeline_start).isoformat(),
                    "total_characters": len(clusters),
                    "processed": idx,
                    "results": all_results
                }, f, indent=2)

        # Summary
        pipeline_elapsed = time.time() - pipeline_start

        self.log(f"\n{'='*80}")
        self.log(f"PIPELINE COMPLETE")
        self.log(f"{'='*80}")
        self.log(f"Total time: {pipeline_elapsed/3600:.2f} hours")
        self.log(f"Characters processed: {len(all_results)}")

        completed = sum(1 for r in all_results if r["status"] == "completed")
        self.log(f"Successful: {completed}")
        self.log(f"Failed: {len(all_results) - completed}")

        self.log(f"\nResults: {self.base_output_dir / 'pipeline_results.json'}")
        self.log(f"Log: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Yokai Watch Optimized Pipeline - Uses ALL frames"
    )

    parser.add_argument(
        "--input-frames",
        type=Path,
        default=Path("/home/b0979/yokai_input_fast"),
        help="Directory with ALL raw frames"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/optimized_pipeline"),
        help="Output directory"
    )
    parser.add_argument(
        "--segmentation-sample",
        type=int,
        default=10,
        help="Number of episodes to sample for segmentation (default: 10)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=50,
        help="Minimum cluster size (using more frames, so increase threshold)"
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=None,
        help="Max characters to train (None for all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors",
        help="Base SD model"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)

    # Create pipeline
    pipeline = OptimizedYokaiPipeline(
        input_frames_dir=args.input_frames,
        base_output_dir=args.output_dir,
        segmentation_sample_size=args.segmentation_sample,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        base_model=args.base_model
    )

    # Run
    pipeline.run(max_characters=args.max_characters)


if __name__ == "__main__":
    main()
