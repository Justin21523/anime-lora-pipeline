#!/usr/bin/env python3
"""
Stage 2a: Character Refinement with U²-Net and MODNet
Refines character boundaries from Stage 1 coarse masks
Handles fine details like hair strands, clothing edges, and transparent regions
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

# Try to import U2NET
try:
    from u2net import U2NET, U2NETP
    U2NET_AVAILABLE = True
except ImportError:
    U2NET_AVAILABLE = False
    print("⚠️  U2NET not installed. Install with: pip install git+https://github.com/xuebinqin/U-2-Net.git")

# Try to import MODNet
try:
    from modnet import MODNet
    MODNET_AVAILABLE = True
except ImportError:
    MODNET_AVAILABLE = False
    print("⚠️  MODNet not installed. Install with: pip install git+https://github.com/ZHKKKe/MODNet.git")


class CharacterRefiner:
    """
    Character boundary refinement using U²-Net and MODNet
    Specializes in anime character edge detection and matting
    """
    
    def __init__(
        self,
        model_type: str = 'u2net',  # 'u2net' or 'modnet'
        weights_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Initialize character refiner
        
        Args:
            model_type: Type of model to use ('u2net' or 'modnet')
            weights_path: Path to pretrained weights
            device: Device for inference
        """
        self.device = device
        self.model_type = model_type
        
        print(f"🔄 Loading {model_type.upper()} model...")
        
        if model_type == 'u2net':
            if not U2NET_AVAILABLE:
                raise ImportError("U2NET not available")
            
            # Load U2NET model
            self.model = U2NET(3, 1)  # 3 input channels, 1 output channel
            
            if weights_path:
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
            
            self.model = self.model.to(device)
            self.model.eval()
            
        elif model_type == 'modnet':
            if not MODNET_AVAILABLE:
                raise ImportError("MODNet not available")
            
            # Load MODNet model
            self.model = MODNet(backbone_pretrained=False)
            
            if weights_path:
                self.model = torch.nn.DataParallel(self.model).to(device)
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
            else:
                self.model = self.model.to(device)
            
            self.model.eval()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"✓ {model_type.upper()} model loaded")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert to tensor
        if self.model_type == 'u2net':
            # U2NET expects normalized [0, 1]
            tensor = torch.from_numpy(image).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            
        elif self.model_type == 'modnet':
            # MODNet expects normalized [0, 1]
            tensor = torch.from_numpy(image).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def refine_mask(
        self,
        image: np.ndarray,
        coarse_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Refine character mask with fine boundary details
        
        Args:
            image: RGB image (H, W, 3)
            coarse_mask: Optional coarse mask from Stage 1 (H, W)
            
        Returns:
            Refined alpha matte (H, W) with values [0, 255]
        """
        h, w = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run model
        if self.model_type == 'u2net':
            # U2NET outputs 7 side outputs, use the first one
            d1, *_ = self.model(input_tensor)
            pred = torch.sigmoid(d1[0, 0]).cpu().numpy()
            
        elif self.model_type == 'modnet':
            # MODNet outputs alpha matte
            _, _, pred = self.model(input_tensor, True)
            pred = pred[0, 0].cpu().numpy()
        
        # Resize to original size
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to [0, 255]
        pred = (pred * 255).astype(np.uint8)
        
        # If coarse mask provided, use it as guidance
        if coarse_mask is not None:
            # Combine: take maximum confidence
            coarse_mask_norm = coarse_mask.astype(np.float32) / 255.0
            pred_norm = pred.astype(np.float32) / 255.0
            
            # Weighted combination
            combined = pred_norm * 0.7 + coarse_mask_norm * 0.3
            pred = (combined * 255).astype(np.uint8)
        
        return pred
    
    def extract_character_rgba(
        self,
        image: np.ndarray,
        alpha_matte: np.ndarray
    ) -> np.ndarray:
        """
        Extract character with alpha channel
        
        Args:
            image: RGB image (H, W, 3)
            alpha_matte: Alpha matte (H, W)
            
        Returns:
            RGBA image (H, W, 4)
        """
        # Create RGBA
        rgba = np.dstack([image, alpha_matte])
        return rgba


def process_frames(
    input_dir: Path,
    output_dir: Path,
    coarse_masks_dir: Optional[Path] = None,
    model_type: str = 'u2net',
    weights_path: Optional[str] = None,
    device: str = 'cuda'
):
    """
    Process frames with character refinement
    
    Args:
        input_dir: Directory containing frame images
        output_dir: Directory to save refined results
        coarse_masks_dir: Optional directory with Stage 1 coarse masks
        model_type: Model type ('u2net' or 'modnet')
        weights_path: Path to model weights
        device: Device for inference
    """
    print(f"\n{'='*80}")
    print("🎨 Stage 2a: Character Boundary Refinement")
    print(f"{'='*80}\n")
    
    # Create refiner
    refiner = CharacterRefiner(
        model_type=model_type,
        weights_path=weights_path,
        device=device
    )
    
    # Collect frame paths
    print("📂 Scanning for frames...")
    frame_paths = sorted(list(input_dir.rglob('*.jpg')) + list(input_dir.rglob('*.png')))
    frame_paths = [p for p in frame_paths if '_mask' not in p.name and '_character' not in p.name]
    
    print(f"✓ Found {len(frame_paths)} frames\n")
    
    if len(frame_paths) == 0:
        print("⚠️  No frames found!")
        return
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    refined_masks_dir = output_dir / 'refined_masks'
    characters_dir = output_dir / 'characters_rgba'
    refined_masks_dir.mkdir(exist_ok=True)
    characters_dir.mkdir(exist_ok=True)
    
    # Process
    print("🎨 Refining character boundaries...\n")
    
    for frame_path in tqdm(frame_paths, desc="Refining", unit="frame"):
        # Load image
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load coarse mask if available
        coarse_mask = None
        if coarse_masks_dir:
            mask_name = frame_path.stem + '_mask.png'
            mask_path = coarse_masks_dir / mask_name
            if mask_path.exists():
                coarse_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Refine
        refined_alpha = refiner.refine_mask(image_rgb, coarse_mask)
        
        # Save refined mask
        mask_output = refined_masks_dir / f"{frame_path.stem}_refined_mask.png"
        cv2.imwrite(str(mask_output), refined_alpha)
        
        # Extract RGBA character
        character_rgba = refiner.extract_character_rgba(image_rgb, refined_alpha)
        
        # Save character
        char_output = characters_dir / f"{frame_path.stem}_character.png"
        cv2.imwrite(str(char_output), cv2.cvtColor(character_rgba, cv2.COLOR_RGBA2BGRA))
    
    print(f"\n✅ Character refinement complete!")
    print(f"Results saved to: {output_dir}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Stage 2a: Character Boundary Refinement'
    )
    parser.add_argument('input_dir', type=Path, help='Input directory with frames')
    parser.add_argument('-o', '--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--coarse-masks', type=Path, help='Directory with Stage 1 coarse masks')
    parser.add_argument('--model', choices=['u2net', 'modnet'], default='u2net', help='Model type')
    parser.add_argument('--weights', type=str, help='Path to model weights')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device')
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    process_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        coarse_masks_dir=args.coarse_masks,
        model_type=args.model,
        weights_path=args.weights,
        device=args.device
    )


if __name__ == '__main__':
    main()
