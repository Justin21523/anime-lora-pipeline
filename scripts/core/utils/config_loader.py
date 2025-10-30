"""
Configuration loader utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    # Go up: config_loader.py -> utils -> core -> scripts -> project_root
    return current.parent.parent.parent.parent


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_name: str = "global_config") -> OmegaConf:
    """
    Load configuration file

    Args:
        config_name: Config file name (without .yaml extension)

    Returns:
        OmegaConf configuration object
    """
    project_root = get_project_root()
    config_path = project_root / "config" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Convert relative paths to absolute
    if hasattr(config, "paths"):
        for key, value in config.paths.items():
            if isinstance(value, str) and not Path(value).is_absolute():
                config.paths[key] = str(project_root / value)

    return config


def load_character_config(character_name: str) -> OmegaConf:
    """
    Load character-specific configuration

    Args:
        character_name: Character name (e.g., 'endou_mamoru')

    Returns:
        OmegaConf configuration object
    """
    project_root = get_project_root()
    config_path = project_root / "config" / "characters" / f"{character_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Character config not found: {config_path}")

    char_config = OmegaConf.load(config_path)

    # Also load global config and merge
    global_config = load_config("global_config")

    # Merge configs (character config takes precedence)
    merged_config = OmegaConf.merge(global_config, {"character_config": char_config})

    return merged_config


def save_config(config: OmegaConf, output_path: Path):
    """Save configuration to YAML file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


if __name__ == "__main__":
    # Test loading configs
    print("Testing config loader...")

    global_cfg = load_config()
    print(f"✓ Global config loaded")
    print(f"  Project: {global_cfg.project.name}")
    print(f"  GPU: {global_cfg.hardware.gpu.device}")

    char_cfg = load_character_config("endou_mamoru")
    print(f"\n✓ Character config loaded")
    print(f"  Character: {char_cfg.character_config.character.name}")
    print(f"  Trigger word: {char_cfg.character_config.character.trigger_word}")

    print("\n✓ All configs loaded successfully!")
