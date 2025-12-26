"""Configuration utilities using Hydra."""

from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
import hydra


def load_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """Load configuration from YAML file."""
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, path: str):
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        OmegaConf.save(config, f)


def merge_configs(base: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    """Merge base config with overrides."""
    return OmegaConf.merge(base, OmegaConf.create(overrides))
