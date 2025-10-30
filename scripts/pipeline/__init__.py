"""
Professional Anime Video Processing Pipeline
Multi-stage deep learning pipeline for anime frame segmentation, refinement, and analysis
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

# Pipeline stages
STAGES = [
    "stage1_segmentation",       # Mask2Former base segmentation
    "stage2a_character_refine",  # U²-Net + MODNet character refinement
    "stage2b_effect_separation", # CLIPSeg + Grounded-SAM effect extraction
    "stage2c_background_inpaint",# LaMa + Video Inpainting background completion
    "stage3_temporal_consistency",# XMem + DeAOT temporal smoothing
    "stage4_annotation",          # CLIP + BLIP-2 + DINOv2 intelligent annotation
]
