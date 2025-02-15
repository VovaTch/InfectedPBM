from __future__ import annotations

import hydra
from omegaconf import DictConfig


def load_cfg_from_hydra(
    config_path: str, config_name: str, overrides: list[str] | None = None
) -> DictConfig:
    """
    Load a configuration from Hydra.

    Args:
        config_path (str): The path to the configuration directory.
        config_name (str): The name of the configuration file.

    Returns:
        DictConfig: The configuration as a DictConfig.
    """
    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    return cfg
