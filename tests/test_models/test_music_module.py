import torch
import torch.nn as nn
import pytest
from omegaconf import DictConfig

from loss.aggregators import LossOutput
from models.base import BaseLightningModule
from models.multi_level_vqvae.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from models.music_module import MusicLightningModule
from utils.containers import LearningParameters


@pytest.fixture
def music_module(cfg: DictConfig) -> MusicLightningModule:
    learning_params = LearningParameters.from_cfg(cfg)
    model = MultiLvlVQVariationalAutoEncoder.from_cfg(cfg)
    return MusicLightningModule(model, learning_params)


def test_MusicLightningModule_forward(music_module: MusicLightningModule) -> None:
    input_tensor = {"slice": torch.randn(1, 1, 512)}
    output_tensor = music_module.forward(input_tensor)
    assert isinstance(output_tensor, dict)
    assert "output" in output_tensor
    assert output_tensor["output"].shape == (1, 1, 512)


def test_MusicLightningModule_handle_loss(music_module: MusicLightningModule) -> None:
    loss = LossOutput(
        individuals={"loss1": torch.tensor(0.5), "loss2": torch.tensor(0.3)},
        total=torch.tensor(0.8),
    )
    phase = "train"
    total_loss = music_module.handle_loss(loss, phase)
    assert total_loss == loss.total


def test_MusicLightningModule_from_cfg(cfg: DictConfig) -> None:
    music_module_instance = MusicLightningModule.from_cfg(cfg)
    assert isinstance(music_module_instance, MusicLightningModule)
    assert isinstance(music_module_instance.model, MultiLvlVQVariationalAutoEncoder)
    assert isinstance(music_module_instance.learning_params, LearningParameters)


def test_build_optimizer_adamw(cfg: DictConfig) -> None:
    """
    Test the _build_optimizer method with AdamW.
    """
    optimizer_cfg = {
        "type": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "amsgrad": False,
    }
    model = nn.Linear(10, 1)
    learning_params = LearningParameters.from_cfg(cfg)
    model = BaseLightningModule(
        model, learning_params, optimizer_cfg=optimizer_cfg, scheduler_cfg=None
    )
    assert isinstance(model.optimizer, torch.optim.AdamW)


def test_build_optimizer_sgd(cfg: DictConfig) -> None:
    """
    Test the _build_optimizer method with SGD.
    """
    optimizer_cfg = {
        "type": "SGD",
        "lr": 0.001,
        "momentum": 0.9,
        "dampening": 0,
        "weight_decay": 0.0001,
        "nesterov": False,
    }
    model = nn.Linear(10, 1)
    learning_params = LearningParameters.from_cfg(cfg)
    model = BaseLightningModule(
        model, learning_params, optimizer_cfg=optimizer_cfg, scheduler_cfg=None
    )
    assert isinstance(model.optimizer, torch.optim.SGD)


def test_build_scheduler_StepLR(cfg: DictConfig) -> None:
    """
    Test the _build_scheduler method with StepLR.
    """
    scheduler_cfg = {"type": "StepLR", "step_size": 1, "gamma": 0.1}
    model = nn.Linear(10, 1)
    learning_params = LearningParameters.from_cfg(cfg)
    model = BaseLightningModule(
        model, learning_params, optimizer_cfg=None, scheduler_cfg=scheduler_cfg
    )
    assert isinstance(model.scheduler, torch.optim.lr_scheduler.StepLR)


def test_build_scheduler_MultiStepLR(cfg: DictConfig) -> None:
    """
    Test the _build_scheduler method with MultiStepLR.
    """
    scheduler_cfg = {"type": "MultiStepLR", "milestones": [1, 2], "gamma": 0.1}
    model = nn.Linear(10, 1)
    learning_params = LearningParameters.from_cfg(cfg)
    model = BaseLightningModule(
        model, learning_params, optimizer_cfg=None, scheduler_cfg=scheduler_cfg
    )
    assert isinstance(model.scheduler, torch.optim.lr_scheduler.MultiStepLR)
