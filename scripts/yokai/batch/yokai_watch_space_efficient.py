#!/usr/bin/env python3
"""
Yokai Watch Space-Efficient Pipeline
Process episodes in batches, clean up intermediate files to save space
Only keep final character clusters and training data
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


class SpaceEfficientPipeline:
    """Space-efficient pipeline that processes and cleans up as it goes"""

    def __init__(
        self,
        input_frames_dir: Path,
        base_output_dir: Path,
        episodes_per_batch: int = 10,
        min_cluster_size: int = 25,
        device: str = "cuda",
        base_model: str = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors",
        temp_dir: Optional[Path] = None
    ):
        """
        Initialize space-efficient pipeline

        Args:
            input_frames_dir: Directory with raw frames (e.g., /home/b0979/yokai_input_fast)
            base_output_dir: Base directory for outputs
            episodes_per_batch: Process this many episodes at a time
            min_cluster_size: Minimum cluster size
            device: Device to use
            base_model: Base SD model path
        """
        self.input_frames_dir = Path(input_frames_dir)
        self.base_output_dir = Path(base_output_dir)
        self.episodes_per_batch = episodes_per_batch
        self.min_cluster_size = min_cluster_size
        self.device = device
        self.base_model = base_model

        # Setup paths
        if temp_dir is None:
            import time
            timestamp = int(time.time())
            self.temp_dir = Path(f"/tmp/yokai_processing_{timestamp}")
        else:
            self.temp_dir = Path(temp_dir)
        self.temp_layered = self.temp_dir / "layered_frames"
        self.temp_clusters = self.temp_dir / "clusters"

        # Final output directories
        self.warehouse_training = Path("/mnt/c/AI_LLM_projects/ai_warehouse/training_data/yokai-watch")
        self.warehouse_outputs = Path("/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch")
        self.warehouse_models = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/yokai-watch")

        # Script paths
        self.project_root = Path("/mnt/c/AI_LLM_projects/inazuma-eleven-lora")
        self.segmentation_script = self.project_root / "scripts/tools/layered_segmentation.py"
        self.clustering_script = self.project_root / "scripts/tools/character_clustering.py"
        self.ai_analysis_script = self.project_root / "scripts/evaluation/comprehensive_anime_analysis.py"
        self.caption_apply_script = self.project_root / "scripts/tools/apply_captions_from_analysis.py"
        self.test_script = self.project_root / "scripts/evaluation/test_lora_checkpoints.py"

        # Conda environment
        self.conda_env = "blip2-env"

        # Create output directories
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.warehouse_training.mkdir(parents=True, exist_ok=True)
        self.warehouse_outputs.mkdir(parents=True, exist_ok=True)
        self.warehouse_models.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_file = self.base_output_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        self.log(f"{'='*80}")
        self.log(f"Yokai Watch Space-Efficient Pipeline")
        self.log(f"{'='*80}")
        self.log(f"Input Frames: {self.input_frames_dir}")
        self.log(f"Output Base: {self.base_output_dir}")
        self.log(f"Episodes per Batch: {self.episodes_per_batch}")
        self.log(f"Min Cluster Size: {self.min_cluster_size}")
        self.log(f"Device: {self.device}")
        self.log(f"Temp Directory: {self.temp_dir}")
        self.log(f"{'='*80}\n")

    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and log output"""
        self.log(f"\n{'='*80}")
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(cmd)}")
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

    def cleanup_temp(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            self.log(f"🧹 Cleaning up temp directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.log("✓ Temp directory cleaned")

    def get_episode_folders(self) -> List[Path]:
        """Get list of episode folders"""
        episodes = sorted([d for d in self.input_frames_dir.iterdir() if d.is_dir()])
        self.log(f"📺 Found {len(episodes)} episode folders")
        return episodes

    def process_batch_segmentation(self, episode_folders: List[Path]) -> bool:
        """Process a batch of episodes for segmentation"""
        self.log(f"\n🎨 Processing Segmentation for {len(episode_folders)} episodes")

        # Create temp directory
        self.temp_layered.mkdir(parents=True, exist_ok=True)

        # Process each episode
        for ep_folder in episode_folders:
            self.log(f"  Segmenting: {ep_folder.name}")

            cmd = [
                "conda", "run", "-n", self.conda_env,
                "python3", str(self.segmentation_script),
                str(ep_folder),
                "--output-dir", str(self.temp_layered),
                "--device", self.device
            ]

            success = self.run_command(cmd, f"Segment {ep_folder.name}")
            if not success:
                self.log(f"⚠️  Segmentation failed for {ep_folder.name}")

        return True

    def process_batch_clustering(self) -> bool:
        """Cluster characters from current batch"""
        self.log(f"\n👥 Clustering Characters from Batch")

        # Create temp clusters directory
        self.temp_clusters.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.clustering_script),
            str(self.temp_layered),
            "--output-dir", str(self.temp_clusters),
            "--min-cluster-size", str(self.min_cluster_size),
            "--device", self.device,
            "--copy"
        ]

        return self.run_command(cmd, "Batch Clustering")

    def merge_clusters_to_final(self) -> bool:
        """Merge batch clusters into final character folders"""
        self.log(f"\n🔀 Merging Clusters to Final Output")

        if not self.temp_clusters.exists():
            self.log("⚠️  No temp clusters to merge")
            return False

        # Get batch clusters
        batch_clusters = [d for d in self.temp_clusters.iterdir() if d.is_dir() and d.name.startswith("cluster_")]

        self.log(f"Found {len(batch_clusters)} clusters in batch")

        for cluster_dir in batch_clusters:
            cluster_name = cluster_dir.name
            final_cluster_dir = self.warehouse_training / cluster_name

            # Create final cluster directory if it doesn't exist
            final_cluster_dir.mkdir(parents=True, exist_ok=True)

            # Copy files from batch cluster to final cluster
            cluster_images = list(cluster_dir.glob("*.png"))
            self.log(f"  Merging {cluster_name}: {len(cluster_images)} images")

            for img_path in cluster_images:
                # Create unique filename to avoid overwriting
                new_name = f"batch_{img_path.name}"
                dest_path = final_cluster_dir / new_name

                # Copy if not exists
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)

        self.log("✓ Clusters merged")
        return True

    def get_final_clusters(self) -> List[Path]:
        """Get list of final character cluster directories"""
        clusters = [d for d in self.warehouse_training.iterdir() if d.is_dir() and d.name.startswith("cluster_")]
        clusters.sort()

        self.log(f"\n📊 Final Character Clusters: {len(clusters)}")
        for cluster in clusters:
            num_images = len(list(cluster.glob("*.png")))
            self.log(f"  - {cluster.name}: {num_images} images")

        return clusters

    def process_character_analysis(self, cluster_dir: Path) -> bool:
        """AI analysis for a character cluster"""
        self.log(f"\n🤖 AI Analysis for {cluster_dir.name}")

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

    def process_character_captions(self, cluster_dir: Path) -> bool:
        """Apply captions for a character cluster"""
        self.log(f"\n📝 Apply Captions for {cluster_dir.name}")

        analysis_output = self.warehouse_outputs / cluster_dir.name / "ai_analysis"
        analysis_json = analysis_output / "character_analysis.json"

        if not analysis_json.exists():
            self.log(f"⚠️  Analysis JSON not found: {analysis_json}")
            return False

        # Apply captions (in-place in the cluster directory)
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.caption_apply_script),
            str(analysis_json),
            str(cluster_dir),
            "--output-dir", str(cluster_dir)  # Apply captions in same directory
        ]

        return self.run_command(cmd, f"Apply Captions for {cluster_dir.name}")

    def train_character_lora(self, cluster_dir: Path) -> bool:
        """Train LoRA for a character"""
        character_name = cluster_dir.name
        self.log(f"\n🎓 Training LoRA for {character_name}")

        # Check training data
        num_images = len(list(cluster_dir.glob("*.png")))
        num_captions = len(list(cluster_dir.glob("*.txt")))

        self.log(f"Training data: {num_images} images, {num_captions} captions")

        if num_images < self.min_cluster_size or num_captions == 0:
            self.log(f"⚠️  Insufficient training data for {character_name}")
            return False

        # Output directory for this training
        model_output_dir = self.warehouse_models / cluster_dir.name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Training command
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
            "--max_train_steps", "2000",
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

    def test_character_lora(self, cluster_dir: Path) -> bool:
        """Test LoRA checkpoints"""
        self.log(f"\n🧪 Testing LoRA for {cluster_dir.name}")

        model_dir = self.warehouse_models / cluster_dir.name
        test_output_dir = self.warehouse_outputs / cluster_dir.name / "lora_tests"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = list(model_dir.glob("*.safetensors"))
        if not checkpoints:
            self.log(f"⚠️  No checkpoints found in {model_dir}")
            return False

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.test_script),
            str(model_dir),
            "--base-model", self.base_model,
            "--output-dir", str(test_output_dir),
            "--device", self.device
        ]

        return self.run_command(cmd, f"LoRA Testing for {cluster_dir.name}")

    def run(self, total_episodes: Optional[int] = None, max_characters: Optional[int] = None):
        """
        Run the space-efficient pipeline

        Args:
            total_episodes: Total episodes to process (None for all)
            max_characters: Max characters to train (None for all)
        """
        pipeline_start = time.time()

        # Get all episode folders
        all_episodes = self.get_episode_folders()

        if total_episodes:
            all_episodes = all_episodes[:total_episodes]
            self.log(f"Processing first {total_episodes} episodes")

        # Process episodes in batches
        num_batches = (len(all_episodes) + self.episodes_per_batch - 1) // self.episodes_per_batch

        self.log(f"\n📦 Processing {len(all_episodes)} episodes in {num_batches} batch(es)")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.episodes_per_batch
            end_idx = min(start_idx + self.episodes_per_batch, len(all_episodes))
            batch_episodes = all_episodes[start_idx:end_idx]

            self.log(f"\n{'#'*80}")
            self.log(f"BATCH {batch_idx + 1}/{num_batches}")
            self.log(f"Episodes: {batch_episodes[0].name} to {batch_episodes[-1].name}")
            self.log(f"{'#'*80}")

            # Clean previous batch temp files
            self.cleanup_temp()

            # 1. Segmentation for this batch
            if not self.process_batch_segmentation(batch_episodes):
                self.log(f"⚠️  Batch {batch_idx + 1} segmentation had issues")

            # 2. Clustering for this batch
            if not self.process_batch_clustering():
                self.log(f"⚠️  Batch {batch_idx + 1} clustering failed")
                continue

            # 3. Merge batch clusters to final output
            self.merge_clusters_to_final()

            # Clean up this batch's temp files
            self.cleanup_temp()

        # Now process all final character clusters
        self.log(f"\n{'#'*80}")
        self.log(f"CHARACTER PROCESSING")
        self.log(f"{'#'*80}")

        final_clusters = self.get_final_clusters()

        if max_characters:
            final_clusters = final_clusters[:max_characters]
            self.log(f"Processing first {max_characters} character(s)")

        all_results = []

        for idx, cluster_dir in enumerate(final_clusters, 1):
            self.log(f"\n{'='*80}")
            self.log(f"Character {idx}/{len(final_clusters)}: {cluster_dir.name}")
            self.log(f"{'='*80}")

            result = {
                "character": cluster_dir.name,
                "start_time": datetime.now().isoformat(),
                "stages": {}
            }

            start_time = time.time()

            # AI Analysis
            success = self.process_character_analysis(cluster_dir)
            result["stages"]["ai_analysis"] = success
            if not success:
                result["status"] = "failed_analysis"
                all_results.append(result)
                continue

            # Apply Captions
            success = self.process_character_captions(cluster_dir)
            result["stages"]["apply_captions"] = success
            if not success:
                result["status"] = "failed_captions"
                all_results.append(result)
                continue

            # Train LoRA
            success = self.train_character_lora(cluster_dir)
            result["stages"]["train_lora"] = success
            if not success:
                result["status"] = "failed_training"
                all_results.append(result)
                continue

            # Test LoRA
            success = self.test_character_lora(cluster_dir)
            result["stages"]["test_lora"] = success
            result["status"] = "completed" if success else "failed_testing"

            elapsed = time.time() - start_time
            result["end_time"] = datetime.now().isoformat()
            result["elapsed_seconds"] = elapsed

            all_results.append(result)

            # Save intermediate results
            results_file = self.base_output_dir / "pipeline_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({
                    "pipeline_start": datetime.fromtimestamp(pipeline_start).isoformat(),
                    "total_characters": len(final_clusters),
                    "processed": idx,
                    "results": all_results
                }, f, indent=2)

        # Final cleanup
        self.cleanup_temp()

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
        description="Yokai Watch Space-Efficient Pipeline"
    )

    parser.add_argument(
        "--input-frames",
        type=Path,
        default=Path("/home/b0979/yokai_input_fast"),
        help="Directory with raw frames"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch/pipeline"),
        help="Base output directory"
    )
    parser.add_argument(
        "--episodes-per-batch",
        type=int,
        default=10,
        help="Process this many episodes at a time"
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        help="Total episodes to process (None for all)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=25,
        help="Minimum cluster size"
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
        help="Base SD model path"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = SpaceEfficientPipeline(
        input_frames_dir=args.input_frames,
        base_output_dir=args.output_dir,
        episodes_per_batch=args.episodes_per_batch,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        base_model=args.base_model
    )

    # Run pipeline
    pipeline.run(
        total_episodes=args.total_episodes,
        max_characters=args.max_characters
    )


if __name__ == "__main__":
    main()
