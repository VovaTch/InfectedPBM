import torch
import torch.nn as nn

from utils.positional_encoding import SinusoidalPositionEmbeddings, apply_pos_encoding


class AttentionEmbeddingEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feature_extractor: nn.Module,
        class_head: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Parameters
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout = dropout

        # Feature extractor and head
        self._feature_extractor = feature_extractor
        self._class_head = class_head

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the network

        Args:
            x (torch.Tensor): Input sequence, shape (batch_size, input_channels, seq_len)

        Returns:
            torch.Tensor: Output logits tensor, shape (batch_size, num_classes)
        """
        x = self._feature_extractor(x)
        x = x.transpose(1, 2).contiguous()
        x = apply_pos_encoding(x, self._positional_embeddings)
        x = torch.cat((x, self._class_token.repeat(x.shape[0], 1, 1)), dim=1)
        x = self._transformer_encoder(x)
        x = self._class_head(x[:, -1, :])
        return x
