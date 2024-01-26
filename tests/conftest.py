import pytest
from omegaconf import DictConfig
from hydra import initialize, compose


@pytest.fixture()
def cfg() -> DictConfig:
    """
    Fixture that returns the configuration as a DictConfig.

    This fixture initializes the Hydra config directory and composes the configuration
    based on the specified config name. The configuration is returned as a DictConfig.

    Returns:
        The configuration as a DictConfig.
    """
    # Initialize the Hydra config directory
    with initialize(version_base=None, config_path="test_config"):
        # config is relative to a module
        cfg = compose(config_name="test_config.yaml")

    # Return the configuration as a DictConfig
    return cfg


@pytest.fixture
def real_cfg() -> DictConfig:
    """
    Fixture that returns the configuration as a DictConfig.

    This fixture initializes the Hydra config directory and composes the configuration
    based on the specified config name. The configuration is returned as a DictConfig.

    Returns:
        The configuration as a DictConfig.
    """
    # Initialize the Hydra config directory
    with initialize(version_base=None, config_path="../config"):
        # config is relative to a module
        cfg = compose(config_name="config.yaml")

    # Return the configuration as a DictConfig
    return cfg
