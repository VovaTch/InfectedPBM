import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Discriminator, DiscriminatorHead
from utils.positional_encoding import SinusoidalPositionEmbeddings, apply_pos_encoding


class AttentionDiscriminator(Discriminator):
    """
    A discriminator network based on the transformer architecture.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feature_extractor: nn.Module,
        class_head: DiscriminatorHead,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the attention body for the discriminator model.

        Args:
            hidden_dim (int): The dimension of the hidden layers.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of transformer encoder layers.
            feature_extractor (nn.Module): The feature extractor module.
            class_head (DiscriminatorHead): The classification head module.
            dropout (float, optional): The dropout rate. Default is 0.1.

        Returns:
            None
        """
        super().__init__()

        # Parameters
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout = dropout

        # Feature extractor and head
        self._feature_extractor = feature_extractor
        self.class_head = class_head

        # Attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self._transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self._positional_embeddings = SinusoidalPositionEmbeddings(hidden_dim)
        self._class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward method of the network

        Args:
            x (torch.Tensor): Input sequence, shape (batch_size, input_channels, seq_len)

        Returns:
            dict[str, torch.Tensor]: Output logits tensor, shape (batch_size, num_classes)
        """
        x = self._feature_extractor(x)
        x = x.transpose(1, 2).contiguous()
        x = apply_pos_encoding(x, self._positional_embeddings)
        x = torch.cat((x, self._class_token.repeat(x.shape[0], 1, 1)), dim=1)
        x = self._transformer_encoder(x)
        x = self.class_head(x[:, -1, :])
        return {"logits": x}


class PatchAttentionDiscriminator(AttentionDiscriminator):
    """
    Patch-based Discriminator classifier, using the same logic as AttentionEmbeddingEncoder, just with
    patches. If the input doesn't fit neatly into patches, it will pad the input to fit the patches.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        patch_size: int,
        feature_extractor: nn.Module,
        class_head: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the attention body for the discriminator model.

        Args:
            hidden_dim (int): The dimension of the hidden layers.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of attention layers.
            patch_size (int): The size of the patches to be processed.
            feature_extractor (nn.Module): The feature extractor module.
            class_head (nn.Module): The classification head module.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super().__init__(
            hidden_dim, num_heads, num_layers, feature_extractor, class_head, dropout
        )
        self._patch_size = patch_size

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward method of the network

        Args:
            x (torch.Tensor): Tensor size (batch_size, input_channels, seq_len)

        Returns:
            dict[str, torch.Tensor]: Output logits tensor, size (batch_size, num_patches, num_classes)
        """
        x = self._feature_extractor(x)
        x = x.transpose(1, 2).contiguous()
        x = apply_pos_encoding(x, self._positional_embeddings)

        num_patches_full = x.shape[1] // self._patch_size
        num_patches_partial = 0 if x.shape[1] % self._patch_size == 0 else 1
        num_total_patches = num_patches_full + num_patches_partial

        # Pad the input tensor to have total full patches
        x = F.pad(x, (0, 0, 0, num_total_patches * self._patch_size - x.shape[1]))
        x = torch.cat(
            (x, self._class_token.repeat(x.shape[0], num_total_patches, 1)),
            dim=1,
        )

        # Generate a mask
        diags = [
            torch.ones(self._patch_size, self._patch_size).to(x.device)
        ] * num_total_patches + [torch.tensor(1).to(x.device)] * num_total_patches
        diag = torch.block_diag(*diags)
        for idx in range(num_total_patches):
            diag[
                idx - num_total_patches,
                idx * self._patch_size : (idx + 1) * self._patch_size,
            ] = 1
            diag[
                idx * self._patch_size : (idx + 1) * self._patch_size,
                idx - num_total_patches,
            ] = 1
        diag = diag.to(torch.bool)

        x = self._transformer_encoder.forward(x, mask=diag)
        x = self.class_head(x[:, -num_total_patches:, :])
        return {"logits": x}
