from typing_extensions import Self

from mamba_ssm import Mamba
from omegaconf import DictConfig
import torch
import torch.nn as nn

from utils.containers import MambaParams
from common import registry


@registry.register_mel_spec_converter("mamba")
class MambaWrapper(nn.Module):
    def __init__(self, mamba_params: MambaParams, vocabulary_size: int) -> None:
        super().__init__()
        self.mamba_params = mamba_params
        self.mamba = Mamba(
            d_model=mamba_params.model_dim,
            d_state=mamba_params.ssm_state_dim,
            d_conv=mamba_params.conv_width,
            expand=mamba_params.expansion,
        )
        self.vocabulary_size = vocabulary_size
        self.in_embedding = nn.Embedding(vocabulary_size + 2, mamba_params.model_dim)
        self.out_projection = nn.Linear(mamba_params.model_dim, vocabulary_size + 2)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        mamba_params = MambaParams.from_cfg(cfg)
        vocabulary_size = cfg.model.vocabulary_size
        return cls(mamba_params, vocabulary_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Input size BS x L
        embedding = self.in_embedding(x)
        mamba_outputs = self.mamba(embedding)
        return {"logits": self.out_projection(mamba_outputs)}  # size BS x L x Voc

    @property
    def model(self) -> Mamba:
        return self.mamba

    def get_last_logits(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        return outputs["logits"][:, -1, :]
