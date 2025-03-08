from mamba_ssm import Mamba
import torch
import torch.nn as nn


class MambaWrapper(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ssm_state_dim: int,
        conv_width: int,
        expansion: int,
        vocabulary_size: int,
    ) -> None:
        super().__init__()

        self._model_dim = model_dim
        self._ssm_state_dim = ssm_state_dim
        self._conv_width = conv_width
        self._expansion = expansion

        self.mamba = Mamba(
            d_model=model_dim,
            d_state=ssm_state_dim,
            d_conv=conv_width,
            expand=expansion,
        )
        self.vocabulary_size = vocabulary_size
        self.in_embedding = nn.Embedding(vocabulary_size + 2, model_dim)
        self.out_projection = nn.Linear(model_dim, vocabulary_size + 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Input size BS x L
        embedding = self.in_embedding(x)
        mamba_outputs = self.mamba(embedding)
        return {
            "pred_logits": self.out_projection(mamba_outputs)
            .transpose(1, 2)
            .contiguous()
        }  # size BS x Voc + 2 x L

    @property
    def model(self) -> Mamba:
        return self.mamba

    def get_last_logits(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        return outputs["pred_logits"][:, :, -1]
