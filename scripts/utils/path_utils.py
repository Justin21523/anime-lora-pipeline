"""
Path and file system utilities
"""

import shutil
from pathlib import Path
from typing import List, Optional
import json


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    # Go up: path_utils.py -> utils -> scripts -> project_root
    return current.parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if not

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_character_paths(character_name: str) -> dict:
    """
    Get standard paths for a character

    Args:
        character_name: Character name (e.g., 'endou_mamoru')

    Returns:
        Dictionary of paths
    """
    project_root = get_project_root()
    char_root = project_root / "data" / "characters" / character_name

    return {
        'root': char_root,
        'gold_standard': char_root / "gold_standard",
        'auto_collected': char_root / "auto_collected",
        'training_ready': char_root / "training_ready",
        'metadata': char_root / "metadata.json"
    }


def list_images(directory: Path, recursive: bool = False) -> List[Path]:
    """
    List all image files in directory

    Args:
        directory: Directory to search
        recursive: Search recursively

    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if recursive:
        image_files = [
            f for f in directory.rglob('*')
            if f.suffix.lower() in image_extensions
        ]
    else:
        image_files = [
            f for f in directory.glob('*')
            if f.suffix.lower() in image_extensions and f.is_file()
        ]

    return sorted(image_files)


def safe_copy(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """
    Safely copy file with error handling

    Args:
        src: Source file
        dst: Destination file
        overwrite: Allow overwriting existing file

    Returns:
        True if successful
    """
    try:
        if dst.exists() and not overwrite:
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True

    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def safe_move(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """
    Safely move file with error handling

    Args:
        src: Source file
        dst: Destination file
        overwrite: Allow overwriting existing file

    Returns:
        True if successful
    """
    try:
        if dst.exists() and not overwrite:
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True

    except Exception as e:
        print(f"Error moving {src} to {dst}: {e}")
        return False


def save_json(data: dict, file_path: Path, indent: int = 2):
    """Save dictionary to JSON file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Path) -> dict:
    """Load JSON file to dictionary"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB"""
    return file_path.stat().st_size / (1024 * 1024)


def cleanup_empty_dirs(root_dir: Path):
    """Remove empty directories recursively"""
    for dirpath in sorted(root_dir.rglob('*'), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            dirpath.rmdir()


if __name__ == "__main__":
    # Test path utilities
    root = get_project_root()
    print(f"Project root: {root}")

    char_paths = get_character_paths("endou_mamoru")
    print(f"\nCharacter paths:")
    for key, path in char_paths.items():
        print(f"  {key}: {path}")

    # Count images
    if char_paths['gold_standard'].exists():
        images = list_images(char_paths['gold_standard'], recursive=True)
        print(f"\nFound {len(images)} images in gold_standard")
