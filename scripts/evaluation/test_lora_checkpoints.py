#!/usr/bin/env python3
"""
Universal LoRA Checkpoint Testing and Evaluation
Automatically tests all checkpoints in a LoRA training output directory
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path
import json
import argparse
from typing import List, Dict
from datetime import datetime
import subprocess
import sys
from tqdm import tqdm


class LoRACheckpointTester:
    """Test and evaluate all LoRA checkpoints"""

    def __init__(
        self,
        base_model_path: str,
        lora_dir: Path,
        output_base_dir: Path,
        device: str = "cuda",
        vae_path: str = None
    ):
        """
        Initialize tester

        Args:
            base_model_path: Path to base Stable Diffusion model
            lora_dir: Directory containing LoRA checkpoints
            output_base_dir: Base directory for test outputs
            device: Device to use (cuda/cpu)
            vae_path: Optional VAE model path
        """
        self.base_model_path = base_model_path
        self.lora_dir = Path(lora_dir)
        self.output_base_dir = Path(output_base_dir)
        self.device = device
        self.vae_path = vae_path

        print(f"🔧 Initializing LoRA Checkpoint Tester")
        print(f"  Base Model: {base_model_path}")
        print(f"  LoRA Dir: {lora_dir}")
        print(f"  Output Dir: {output_base_dir}")
        print(f"  Device: {device}")

    def find_checkpoints(self) -> List[Path]:
        """Find all LoRA checkpoint files"""
        checkpoints = sorted(self.lora_dir.glob("*.safetensors"))
        print(f"\n📦 Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
        return checkpoints

    def load_pipeline(self, lora_path: Path = None):
        """
        Load Stable Diffusion pipeline with optional LoRA

        Args:
            lora_path: Path to LoRA weights (optional)

        Returns:
            Loaded pipeline
        """
        print(f"\n🚀 Loading pipeline...")
        print(f"  Base: {self.base_model_path}")
        if lora_path:
            print(f"  LoRA: {lora_path.name}")

        # Load pipeline
        pipe = StableDiffusionPipeline.from_single_file(
            self.base_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Use DPM++ solver for faster generation
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # Load LoRA if provided
        if lora_path:
            pipe.load_lora_weights(str(lora_path))
            print(f"  ✓ LoRA weights loaded")

        pipe = pipe.to(self.device)
        print(f"  ✓ Pipeline ready on {self.device}")

        return pipe

    def generate_test_images(
        self,
        pipe,
        prompts: List[str],
        output_dir: Path,
        num_variations: int = 4,
        steps: int = 25,
        cfg_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed_base: int = 42
    ) -> Dict:
        """
        Generate test images with given prompts

        Args:
            pipe: Diffusion pipeline
            prompts: List of test prompts
            output_dir: Output directory
            num_variations: Number of variations per prompt
            steps: Number of inference steps
            cfg_scale: Guidance scale
            width: Image width
            height: Image height
            seed_base: Base seed for reproducibility

        Returns:
            Generation metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🎨 Generating test images...")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Variations per prompt: {num_variations}")
        print(f"  Total images: {len(prompts) * num_variations}")
        print(f"  Steps: {steps}, CFG: {cfg_scale}")

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "prompts": prompts,
            "generation_params": {
                "num_variations": num_variations,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "seed_base": seed_base
            },
            "images": []
        }

        img_idx = 0
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
            for var_idx in range(num_variations):
                seed = seed_base + (prompt_idx * num_variations) + var_idx
                generator = torch.Generator(device=self.device).manual_seed(seed)

                # Generate image
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]

                # Save image
                filename = f"img_{img_idx:05d}_p{prompt_idx:03d}_v{var_idx:02d}.png"
                image_path = output_dir / filename
                image.save(image_path)

                # Add to metadata
                metadata["images"].append({
                    "filename": filename,
                    "prompt": prompt,
                    "prompt_idx": prompt_idx,
                    "variation_idx": var_idx,
                    "seed": seed
                })

                img_idx += 1

        # Save metadata
        metadata_path = output_dir / "generation_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ Generated {img_idx} images")
        print(f"✓ Saved to: {output_dir}")

        return metadata

    def run_quality_evaluation(self, test_output_dir: Path) -> Dict:
        """
        Run quality evaluation on generated images

        Args:
            test_output_dir: Directory with generated images

        Returns:
            Evaluation results
        """
        print(f"\n📊 Running quality evaluation...")

        # Path to evaluation script
        eval_script = Path(__file__).parent / "lora_quality_metrics.py"

        if not eval_script.exists():
            print(f"⚠️  Evaluation script not found: {eval_script}")
            return {}

        # Run evaluation
        cmd = [
            sys.executable,
            str(eval_script),
            str(test_output_dir),
            "--device", self.device,
            "--output-json", str(test_output_dir / "quality_evaluation.json")
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)

            # Load results
            eval_json = test_output_dir / "quality_evaluation.json"
            if eval_json.exists():
                with open(eval_json, 'r') as f:
                    return json.load(f)

        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            print(f"STDERR: {e.stderr}")

        return {}

    def test_checkpoint(
        self,
        checkpoint_path: Path,
        prompts: List[str],
        **gen_kwargs
    ) -> Dict:
        """
        Test a single checkpoint

        Args:
            checkpoint_path: Path to checkpoint
            prompts: Test prompts
            **gen_kwargs: Generation parameters

        Returns:
            Test results including quality metrics
        """
        checkpoint_name = checkpoint_path.stem
        output_dir = self.output_base_dir / checkpoint_name

        print(f"\n{'='*80}")
        print(f"Testing Checkpoint: {checkpoint_name}")
        print(f"{'='*80}")

        # Load pipeline with LoRA
        pipe = self.load_pipeline(checkpoint_path)

        # Generate test images
        gen_metadata = self.generate_test_images(
            pipe, prompts, output_dir, **gen_kwargs
        )

        # Free GPU memory
        del pipe
        torch.cuda.empty_cache()

        # Run evaluation
        eval_results = self.run_quality_evaluation(output_dir)

        # Combine results
        results = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_name,
            "output_dir": str(output_dir),
            "generation_metadata": gen_metadata,
            "evaluation_results": eval_results
        }

        return results

    def test_all_checkpoints(
        self,
        prompts: List[str],
        **gen_kwargs
    ) -> List[Dict]:
        """
        Test all checkpoints in LoRA directory

        Args:
            prompts: Test prompts
            **gen_kwargs: Generation parameters

        Returns:
            List of test results for all checkpoints
        """
        checkpoints = self.find_checkpoints()

        if not checkpoints:
            print("❌ No checkpoints found!")
            return []

        all_results = []

        for checkpoint in checkpoints:
            results = self.test_checkpoint(checkpoint, prompts, **gen_kwargs)
            all_results.append(results)

        # Save combined results
        combined_path = self.output_base_dir / "all_checkpoints_results.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ All results saved to: {combined_path}")

        return all_results

    def run_comparison(self):
        """Run comparison between all tested checkpoints"""
        print(f"\n{'='*80}")
        print("Comparing Checkpoints")
        print(f"{'='*80}")

        # Find all quality evaluation files
        eval_dirs = [d for d in self.output_base_dir.iterdir() if d.is_dir()]

        if not eval_dirs:
            print("❌ No evaluation results found!")
            return

        # Path to comparison script
        compare_script = Path(__file__).parent / "compare_lora_models.py"

        if not compare_script.exists():
            print(f"⚠️  Comparison script not found: {compare_script}")
            return

        # Run comparison
        cmd = [
            sys.executable,
            str(compare_script),
            *[str(d) for d in eval_dirs],
            "--output-json", str(self.output_base_dir / "checkpoint_comparison.json"),
            "--output-viz", str(self.output_base_dir / "checkpoint_comparison.png")
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"❌ Comparison failed: {e}")
            print(f"STDERR: {e.stderr}")


def get_default_prompts() -> List[str]:
    """Get default test prompts for Inazuma Eleven"""
    return [
        "inazuma eleven character, soccer player, running with ball, dynamic pose, stadium background",
        "inazuma eleven character, goalkeeper diving to save, action scene, goal post visible",
        "inazuma eleven character, standing portrait, school uniform, confident expression",
        "inazuma eleven character, team celebration, multiple characters, victory pose",
        "inazuma eleven character, special move activation, glowing effects, dramatic lighting",
        "inazuma eleven character, close-up face, determined expression, anime style",
        "inazuma eleven character, training scene, practice field, sweat and effort",
        "inazuma eleven character, casual clothes, outdoor setting, friendly smile",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Test and evaluate all LoRA checkpoints"
    )
    parser.add_argument(
        "lora_dir",
        type=Path,
        help="Directory containing LoRA checkpoint files"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base Stable Diffusion model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base output directory for test results"
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="JSON file with test prompts (one per line)"
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=4,
        help="Number of variations per prompt (default: 4)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps (default: 25)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height (default: 512)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip image generation, only run evaluation/comparison"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation, only generate images"
    )

    args = parser.parse_args()

    # Load prompts
    if args.prompts_file and args.prompts_file.exists():
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = get_default_prompts()
        print(f"Using {len(prompts)} default prompts")

    # Initialize tester
    tester = LoRACheckpointTester(
        base_model_path=args.base_model,
        lora_dir=args.lora_dir,
        output_base_dir=args.output_dir,
        device=args.device
    )

    if not args.skip_generation:
        # Test all checkpoints
        results = tester.test_all_checkpoints(
            prompts=prompts,
            num_variations=args.num_variations,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            width=args.width,
            height=args.height,
            seed_base=args.seed
        )

    if not args.skip_evaluation:
        # Run comparison
        tester.run_comparison()

    print(f"\n{'='*80}")
    print("✅ Testing Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Individual checkpoint results in subdirectories")
    print(f"  - Combined results: all_checkpoints_results.json")
    print(f"  - Comparison: checkpoint_comparison.json")
    print(f"  - Visualization: checkpoint_comparison.png")


if __name__ == "__main__":
    main()
