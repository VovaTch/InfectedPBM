from __future__ import annotations
from typing import Protocol

import omegaconf
import torch
import torch.nn as nn
import hydra
import yaml

from models.modules.base import load_inner_model_state_dict


class Tokenizer(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
    def tokenize(self, x: torch.Tensor) -> torch.Tensor: ...


class AutoModelLoader(nn.Module):
    def __init__(self, config_path: str, weights_path: str | None = None) -> None:
        super().__init__()
        with open(config_path, "r") as f:
            config = omegaconf.OmegaConf.create(yaml.safe_load(f))
        self._module = hydra.utils.instantiate(config.module, _convert_="partial")
        if weights_path:
            self._module = load_inner_model_state_dict(self._module, weights_path)
        self._model: Tokenizer = self._module.model  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        return self._model.tokenize(x)
