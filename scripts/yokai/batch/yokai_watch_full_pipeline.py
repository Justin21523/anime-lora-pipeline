#!/usr/bin/env python3
"""
Yokai Watch Complete Automation Pipeline
Automatically processes yokai-watch data through the entire pipeline:
1. Character Clustering (from existing layered frames or raw frames)
2. AI Analysis and Captioning
3. LoRA Training for each character
4. Testing and Evaluation
5. Repeat for next character

Can run overnight to process multiple characters
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


class YokaiWatchPipeline:
    """Complete automation pipeline for Yokai Watch LoRA training"""

    def __init__(
        self,
        input_frames_dir: Path,
        base_output_dir: Path,
        min_cluster_size: int = 25,
        device: str = "cuda",
        skip_clustering: bool = False,
        skip_segmentation: bool = False,
        base_model: str = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors"
    ):
        """
        Initialize pipeline

        Args:
            input_frames_dir: Directory with raw frames (e.g., /home/b0979/yokai_input_fast)
            base_output_dir: Base directory for all outputs
            min_cluster_size: Minimum cluster size for character grouping
            device: Device to use (cuda/cpu)
            skip_clustering: Skip clustering if already done
            skip_segmentation: Skip segmentation if layered frames exist
            base_model: Path to base SD model
        """
        self.input_frames_dir = Path(input_frames_dir)
        self.base_output_dir = Path(base_output_dir)
        self.min_cluster_size = min_cluster_size
        self.device = device
        self.skip_clustering = skip_clustering
        self.skip_segmentation = skip_segmentation
        self.base_model = base_model

        # Setup paths
        self.warehouse_cache = Path("/mnt/c/AI_LLM_projects/ai_warehouse/cache/yokai-watch")
        self.warehouse_training = Path("/mnt/c/AI_LLM_projects/ai_warehouse/training_data/yokai-watch")
        self.warehouse_outputs = Path("/mnt/c/AI_LLM_projects/ai_warehouse/outputs/yokai-watch")
        self.warehouse_models = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/lora/yokai-watch")

        self.layered_frames_dir = self.warehouse_cache / "layered_frames"
        self.character_clusters_dir = self.warehouse_training / "character_clusters"

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
        self.warehouse_cache.mkdir(parents=True, exist_ok=True)
        self.warehouse_training.mkdir(parents=True, exist_ok=True)
        self.warehouse_outputs.mkdir(parents=True, exist_ok=True)
        self.warehouse_models.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_file = self.base_output_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        self.log(f"{'='*80}")
        self.log(f"Yokai Watch Complete Pipeline")
        self.log(f"{'='*80}")
        self.log(f"Input Frames: {self.input_frames_dir}")
        self.log(f"Output Base: {self.base_output_dir}")
        self.log(f"Min Cluster Size: {self.min_cluster_size}")
        self.log(f"Device: {self.device}")
        self.log(f"Skip Segmentation: {self.skip_segmentation}")
        self.log(f"Skip Clustering: {self.skip_clustering}")
        self.log(f"Base Model: {self.base_model}")
        self.log(f"{'='*80}\n")

    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def run_command(self, cmd: List[str], description: str) -> bool:
        """
        Run a command and log output

        Args:
            cmd: Command as list
            description: Description of the command

        Returns:
            True if successful, False otherwise
        """
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

            # Log output
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

    def stage1_segmentation(self) -> bool:
        """Stage 1: Segment frames into layered format"""
        if self.skip_segmentation:
            self.log("\n⏭️  Skipping segmentation (skip_segmentation=True)")
            if not self.layered_frames_dir.exists():
                self.log(f"⚠️  Warning: layered_frames_dir doesn't exist: {self.layered_frames_dir}")
                return False
            return True

        self.log(f"\n🎨 Stage 1: Layered Segmentation")

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.segmentation_script),
            str(self.input_frames_dir),
            "--output-dir", str(self.layered_frames_dir),
            "--device", self.device
        ]

        return self.run_command(cmd, "Layered Segmentation")

    def stage2_clustering(self) -> bool:
        """Stage 2: Character clustering from layered frames"""
        if self.skip_clustering:
            self.log("\n⏭️  Skipping clustering (skip_clustering=True)")
            if not self.character_clusters_dir.exists():
                self.log(f"⚠️  Warning: character_clusters_dir doesn't exist: {self.character_clusters_dir}")
                return False
            return True

        self.log(f"\n👥 Stage 2: Character Clustering")

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.clustering_script),
            str(self.layered_frames_dir),
            "--output-dir", str(self.character_clusters_dir),
            "--min-cluster-size", str(self.min_cluster_size),
            "--device", self.device,
            "--copy"  # Copy files instead of moving
        ]

        return self.run_command(cmd, "Character Clustering")

    def get_character_clusters(self) -> List[Path]:
        """Get list of character cluster directories"""
        if not self.character_clusters_dir.exists():
            self.log(f"⚠️  Character clusters directory doesn't exist: {self.character_clusters_dir}")
            return []

        clusters = [d for d in self.character_clusters_dir.iterdir() if d.is_dir() and d.name.startswith("cluster_")]
        clusters.sort()

        self.log(f"\n📊 Found {len(clusters)} character cluster(s):")
        for cluster in clusters:
            num_images = len(list(cluster.glob("*.png")))
            self.log(f"  - {cluster.name}: {num_images} images")

        return clusters

    def stage3_ai_analysis(self, cluster_dir: Path) -> bool:
        """Stage 3: AI analysis and caption generation"""
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
        """Stage 4: Apply captions from AI analysis"""
        self.log(f"\n📝 Stage 4: Apply Captions for {cluster_dir.name}")

        analysis_output = self.warehouse_outputs / cluster_dir.name / "ai_analysis"
        analysis_json = analysis_output / "character_analysis.json"

        if not analysis_json.exists():
            self.log(f"⚠️  Analysis JSON not found: {analysis_json}")
            return False

        # Create training data directory for this character
        training_dir = self.warehouse_training / cluster_dir.name
        training_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", str(self.caption_apply_script),
            str(analysis_json),
            str(cluster_dir),
            "--output-dir", str(training_dir)
        ]

        return self.run_command(cmd, f"Apply Captions for {cluster_dir.name}")

    def stage5_train_lora(self, cluster_dir: Path, character_name: Optional[str] = None) -> bool:
        """Stage 5: Train LoRA model"""
        if character_name is None:
            character_name = cluster_dir.name

        self.log(f"\n🎓 Stage 5: LoRA Training for {character_name}")

        training_dir = self.warehouse_training / cluster_dir.name

        # Check if training data exists
        num_images = len(list(training_dir.glob("*.png")))
        num_captions = len(list(training_dir.glob("*.txt")))

        self.log(f"Training data: {num_images} images, {num_captions} captions")

        if num_images == 0 or num_captions == 0:
            self.log(f"⚠️  Insufficient training data for {character_name}")
            return False

        # Output directory for this training
        model_output_dir = self.warehouse_models / cluster_dir.name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Training command using sd-scripts
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python3", "-m", "accelerate.commands.launch",
            "--num_cpu_threads_per_process", "1",
            "train_network.py",
            "--pretrained_model_name_or_path", self.base_model,
            "--train_data_dir", str(training_dir),
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

    def stage6_test_lora(self, cluster_dir: Path) -> bool:
        """Stage 6: Test LoRA checkpoints"""
        self.log(f"\n🧪 Stage 6: LoRA Testing for {cluster_dir.name}")

        model_dir = self.warehouse_models / cluster_dir.name
        test_output_dir = self.warehouse_outputs / cluster_dir.name / "lora_tests"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if checkpoints exist
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

    def process_character(self, cluster_dir: Path) -> Dict:
        """Process a single character through the entire pipeline"""
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

        # Stage 3: AI Analysis
        success = self.stage3_ai_analysis(cluster_dir)
        results["stages"]["ai_analysis"] = success
        if not success:
            self.log(f"⚠️  AI Analysis failed for {character_name}, skipping...")
            results["status"] = "failed_analysis"
            return results

        # Stage 4: Apply Captions
        success = self.stage4_apply_captions(cluster_dir)
        results["stages"]["apply_captions"] = success
        if not success:
            self.log(f"⚠️  Caption application failed for {character_name}, skipping...")
            results["status"] = "failed_captions"
            return results

        # Stage 5: Train LoRA
        success = self.stage5_train_lora(cluster_dir, character_name)
        results["stages"]["train_lora"] = success
        if not success:
            self.log(f"⚠️  LoRA training failed for {character_name}, skipping...")
            results["status"] = "failed_training"
            return results

        # Stage 6: Test LoRA
        success = self.stage6_test_lora(cluster_dir)
        results["stages"]["test_lora"] = success
        if not success:
            self.log(f"⚠️  LoRA testing failed for {character_name}")
            results["status"] = "failed_testing"
        else:
            results["status"] = "completed"

        elapsed = time.time() - start_time
        results["end_time"] = datetime.now().isoformat()
        results["elapsed_seconds"] = elapsed

        self.log(f"\n✅ Character {character_name} processing complete in {elapsed/60:.1f} minutes")
        return results

    def run(self, max_characters: Optional[int] = None):
        """
        Run the complete pipeline

        Args:
            max_characters: Maximum number of characters to process (None for all)
        """
        pipeline_start = time.time()

        # Stage 1: Segmentation (if needed)
        if not self.stage1_segmentation():
            self.log("❌ Pipeline failed at segmentation stage")
            return

        # Stage 2: Clustering (if needed)
        if not self.stage2_clustering():
            self.log("❌ Pipeline failed at clustering stage")
            return

        # Get character clusters
        clusters = self.get_character_clusters()
        if not clusters:
            self.log("❌ No character clusters found")
            return

        # Limit number of characters if specified
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

        # Final summary
        pipeline_elapsed = time.time() - pipeline_start

        self.log(f"\n{'='*80}")
        self.log(f"PIPELINE COMPLETE")
        self.log(f"{'='*80}")
        self.log(f"Total time: {pipeline_elapsed/3600:.2f} hours")
        self.log(f"Characters processed: {len(all_results)}")

        completed = sum(1 for r in all_results if r["status"] == "completed")
        failed = len(all_results) - completed

        self.log(f"Successful: {completed}")
        self.log(f"Failed: {failed}")

        if completed > 0:
            avg_time = sum(r.get("elapsed_seconds", 0) for r in all_results) / len(all_results)
            self.log(f"Average time per character: {avg_time/60:.1f} minutes")

        self.log(f"\nResults saved to: {self.base_output_dir / 'pipeline_results.json'}")
        self.log(f"Log saved to: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Yokai Watch Complete Automation Pipeline"
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
        "--min-cluster-size",
        type=int,
        default=25,
        help="Minimum cluster size for character grouping"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip segmentation if layered frames exist"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering if character clusters exist"
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=None,
        help="Maximum number of characters to process"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors",
        help="Path to base SD model"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = YokaiWatchPipeline(
        input_frames_dir=args.input_frames,
        base_output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        skip_clustering=args.skip_clustering,
        skip_segmentation=args.skip_segmentation,
        base_model=args.base_model
    )

    # Run pipeline
    pipeline.run(max_characters=args.max_characters)


if __name__ == "__main__":
    main()
