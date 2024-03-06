from typing_extensions import Self
from omegaconf import DictConfig
import torch

import torch.nn as nn
from mamba_ssm import Mamba

from utils.containers import MambaParams
from common import registry


class ExpandingMLPHead(nn.Module):
    """
    Expanding MLP heads for the Chaos Hydra class, might be used for something else though.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        activation: str = "relu",
    ) -> None:
        """
        Initializes a MultiHead model.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            hidden_dim (int): The dimension of the hidden layers.
            num_layers (int, optional): The number of layers in the model. Defaults to 4.
            activation (str, optional): The activation function to use. Defaults to "relu".

        Raises:
            ValueError: If an unknown activation function is provided or if num_layers is less than 2.
        """
        super().__init__()

        match activation:
            case "relu":
                self.activation = nn.ReLU()
            case "gelu":
                self.activation = nn.GELU()
            case "leaky_relu":
                self.activation = nn.LeakyReLU()
            case "tanh":
                self.activation = nn.Tanh()
            case _:
                raise ValueError(f"Unknown activation: {activation}")

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        layers = [nn.Linear(in_dim, hidden_dim), self.activation]
        for inner_layer_idx in range(num_layers - 2):
            layers.append(
                nn.Linear(
                    hidden_dim * 2 ** (inner_layer_idx),
                    hidden_dim * 2 ** (inner_layer_idx + 1),
                )
            )
            layers.append(self.activation)
            layers.append(nn.LayerNorm(hidden_dim * 2 ** (inner_layer_idx + 1)))
        layers += [nn.Linear(hidden_dim * 2 ** (num_layers - 2), out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.mlp(x)


@registry.register_model("chaos_hydra")
class ChaosHydra(nn.Module):
    """
    Multi-headed snake, which is a Hydra pretty much :X but it's confusing with Python's Hydra. As such, this
    is a Chaos Hydra, because Heroes and Might and Magic 3... Anyway, it's a multi-headed model fit for
    RQ-VAE.
    """

    def __init__(
        self,
        mamba_params: MambaParams,
        vocabulary_size: int,
        num_heads: int,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        mlp_activation: str = "relu",
    ) -> None:
        """
        Initializes the MultiHead class.

        Args:
            mamba_params (MambaParams): The parameters for the Mamba model.
            vocabulary_size (int): The size of the vocabulary.
            num_heads (int): The number of attention heads.
            mlp_hidden_dim (int): The hidden dimension of the MLP head.
            mlp_num_layers (int): The number of layers in the MLP head.
            mlp_activation (str, optional): The activation function for the MLP head. Defaults to "relu".
        """
        super().__init__()
        self.mamba_params = mamba_params
        self.mamba = Mamba(
            d_model=mamba_params.model_dim * num_heads,
            d_state=mamba_params.ssm_state_dim,
            d_conv=mamba_params.conv_width,
            expand=mamba_params.expansion,
        )
        self.vocabulary_size = vocabulary_size
        self.in_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    vocabulary_size + 2,
                    mamba_params.model_dim,
                )
                for _ in range(num_heads)
            ]
        )
        self.out_projections = nn.ModuleList(
            [
                ExpandingMLPHead(
                    mamba_params.model_dim * num_heads,
                    vocabulary_size + 2,
                    mlp_hidden_dim,
                    mlp_num_layers,
                    mlp_activation,
                )
                for _ in range(num_heads)
            ]
        )

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create an instance of the MultiHead class from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Returns:
            MultiHead: An instance of the MultiHead class.
        """
        mamba_params = MambaParams.from_cfg(cfg)
        vocabulary_size = cfg.model.vocabulary_size
        num_heads = cfg.model.num_heads
        mlp_hidden_dim = cfg.model.mlp_head.hidden_dim
        mlp_num_layers = cfg.model.mlp_head.num_layers
        mlp_activation = cfg.model.mlp_head.activation
        return cls(
            mamba_params,
            vocabulary_size,
            num_heads,
            mlp_hidden_dim,
            mlp_num_layers,
            mlp_activation,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass of the multi-head model.

        Args:
            x (torch.Tensor): Input tensor of size BS x L x num_CB.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the predicted logits.
                - "pred_logits" (torch.Tensor): Predicted logits of size BS x (Voc + 2) x L x num_CB.
        """
        embeddings = [emb(x[:, :, idx]) for idx, emb in enumerate(self.in_embeddings)]
        embeddings = torch.cat(embeddings, dim=-1)  # BS x L x num_CB * model_dim
        mamba_outputs = self.mamba(embeddings)  # BS x L x num_CB * model_dim
        pred_logits = [
            proj(mamba_outputs).transpose(1, 2) for proj in self.out_projections
        ]  # For each, size BS x (Voc + 2) x L
        pred_logits = torch.stack(pred_logits, dim=-1)  # BS x (Voc + 2) x L x num_CB
        return {"pred_logits": pred_logits}

    @property
    def model(self) -> Mamba:
        """
        Returns the Mamba model associated with this MultiHead object.

        Returns:
            Mamba: The Mamba model.
        """
        return self.mamba

    def get_last_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the last predicted logits from the model's forward pass.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The last predicted logits.
        """
        outputs = self.forward(x)
        return outputs["pred_logits"][:, :, -1, :]
