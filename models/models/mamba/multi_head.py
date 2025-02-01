import torch
import torch.nn as nn
from mamba_ssm import Mamba


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
        activation_fn: nn.Module = nn.LeakyReLU(),
    ) -> None:
        """
        Initializes a MultiHead model.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            hidden_dim (int): The dimension of the hidden layers.
            num_layers (int, optional): The number of layers in the model. Defaults to 4.
            activation_fn (module, optional): activation function to use. Defaults to nn.LeakyReLU().

        Raises:
            ValueError: If an unknown activation function is provided or if num_layers is less than 2.
        """
        super().__init__()
        self._activation_fn = activation_fn

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        layers = [nn.Linear(in_dim, hidden_dim), self._activation_fn]
        for inner_layer_idx in range(num_layers - 2):
            layers.append(
                nn.Linear(
                    hidden_dim * 2 ** (inner_layer_idx),
                    hidden_dim * 2 ** (inner_layer_idx + 1),
                )
            )
            layers.append(self._activation_fn)
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


class ChaosHydra(nn.Module):
    """
    Multi-headed snake, which is a Hydra pretty much :X but it's confusing with Python's Hydra. As such, this
    is a Chaos Hydra, because Heroes and Might and Magic 3... Anyway, it's a multi-headed model fit for
    RQ-VAE.
    """

    def __init__(
        self,
        model_dim: int,
        ssm_state_dim: int,
        conv_width: int,
        expansion: int,
        vocabulary_size: int,
        num_heads: int,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        mlp_activation_fn: nn.Module = nn.LeakyReLU(),
    ) -> None:
        """
        Initializes the MultiHead class.

        Args:
            mamba_params (MambaParams): The parameters for the Mamba model.
            vocabulary_size (int): The size of the vocabulary.
            num_heads (int): The number of attention heads.
            mlp_hidden_dim (int): The hidden dimension of the MLP head.
            mlp_num_layers (int): The number of layers in the MLP head.
            mlp_activation_fn (module, optional): activation function to use. Defaults to nn.LeakyReLU().
        """
        super().__init__()
        self.mamba = Mamba(
            d_model=model_dim * num_heads,
            d_state=ssm_state_dim,
            d_conv=conv_width,
            expand=expansion,
        )
        self.vocabulary_size = vocabulary_size
        self.in_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    vocabulary_size + 2,
                    model_dim,
                )
                for _ in range(num_heads)
            ]
        )
        self.out_projections = nn.ModuleList(
            [
                ExpandingMLPHead(
                    model_dim * num_heads,
                    vocabulary_size + 2,
                    mlp_hidden_dim,
                    mlp_num_layers,
                    mlp_activation_fn,
                )
                for _ in range(num_heads)
            ]
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
            proj(mamba_outputs).transpose(1, 2).contiguous()
            for proj in self.out_projections
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
