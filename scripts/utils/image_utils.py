"""
Image processing utilities
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional
import imagehash


def load_image_rgb(image_path: Path) -> np.ndarray:
    """
    Load image as RGB numpy array

    Args:
        image_path: Path to image file

    Returns:
        RGB image as numpy array
    """
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def load_image_cv2(image_path: Path) -> np.ndarray:
    """
    Load image using OpenCV (BGR format)

    Args:
        image_path: Path to image file

    Returns:
        BGR image as numpy array
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img


def is_image_blurry(image_path: Path, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    Check if image is blurry using Laplacian variance

    Args:
        image_path: Path to image file
        threshold: Blur threshold (lower = more blurry)

    Returns:
        Tuple of (is_blurry, blur_score)
    """
    img = load_image_cv2(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var < threshold, float(laplacian_var)


def calculate_brightness(image_path: Path) -> float:
    """
    Calculate average brightness of image

    Args:
        image_path: Path to image file

    Returns:
        Average brightness (0-255)
    """
    img = load_image_cv2(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def get_image_hash(image_path: Path, hash_size: int = 8) -> str:
    """
    Calculate perceptual hash of image for duplicate detection

    Args:
        image_path: Path to image file
        hash_size: Hash size (default 8x8)

    Returns:
        Hex string of image hash
    """
    img = Image.open(image_path)
    return str(imagehash.phash(img, hash_size=hash_size))


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize image to target size

    Args:
        image: Input image
        target_size: (width, height)
        keep_aspect_ratio: Maintain aspect ratio

    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Pad to target size
        if new_w < target_w or new_h < target_h:
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left

            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        return resized
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get image dimensions without loading full image

    Args:
        image_path: Path to image file

    Returns:
        (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def check_image_quality(
    image_path: Path,
    min_resolution: Tuple[int, int] = (512, 512),
    blur_threshold: float = 100.0,
    brightness_range: Tuple[float, float] = (30, 225)
) -> Tuple[bool, dict]:
    """
    Comprehensive image quality check

    Args:
        image_path: Path to image file
        min_resolution: Minimum (width, height)
        blur_threshold: Blur threshold
        brightness_range: (min_brightness, max_brightness)

    Returns:
        Tuple of (passes_quality_check, quality_metrics)
    """
    metrics = {}

    try:
        # Check resolution
        width, height = get_image_dimensions(image_path)
        metrics['width'] = width
        metrics['height'] = height
        metrics['resolution_ok'] = width >= min_resolution[0] and height >= min_resolution[1]

        # Check blur
        is_blurry, blur_score = is_image_blurry(image_path, blur_threshold)
        metrics['blur_score'] = blur_score
        metrics['is_blurry'] = is_blurry

        # Check brightness
        brightness = calculate_brightness(image_path)
        metrics['brightness'] = brightness
        metrics['brightness_ok'] = brightness_range[0] <= brightness <= brightness_range[1]

        # Overall pass
        passes = (
            metrics['resolution_ok'] and
            not metrics['is_blurry'] and
            metrics['brightness_ok']
        )

        return passes, metrics

    except Exception as e:
        metrics['error'] = str(e)
        return False, metrics


if __name__ == "__main__":
    # Test image utilities
    print("Image utilities loaded successfully!")
