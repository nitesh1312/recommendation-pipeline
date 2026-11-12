"""
Configuration Loader Module.
Loads configuration settings from YAML files with safe parsing and path resolution.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Relative or absolute path to the YAML config file.
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    root = get_project_root()
    full_path = root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")
        
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
