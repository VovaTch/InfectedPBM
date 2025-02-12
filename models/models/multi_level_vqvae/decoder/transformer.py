from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.positional_encoding import SinusoidalPositionEmbeddings, apply_pos_encoding


@dataclass
class TransformerParameters:
    """
    A class to hold the parameters for a Transformer model.

    Attributes:
        hidden_dim (int): The dimension of the hidden layers in the Transformer.
        num_encoders (int): The number of encoder layers in the Transformer.
        num_decoders (int): The number of decoder layers in the Transformer.
        num_heads (int): The number of attention heads in each multi-head attention layer.
    """

    hidden_dim: int
    num_encoders: int
    num_decoders: int
    num_heads: int


class TransformerMusicDecoder(nn.Module):
    """
    Transformer encoder-decoder model for music generation out of the latent dim, intended to be used
    as a RQ-VAE decoder.
    """

    def __init__(
        self,
        input_channels: int,
        params: TransformerParameters,
        vocabulary_size: int,
        slice_len: int,
    ) -> None:
        """
        Initializer method

        Args:
            input_channels (int): Input channels from the latent dimension.
            params (TransformerParameters): Parameter object for the transformer architecture.
            vocabulary_size (int): Quantized resolution of each waveform value.
            slice_len (int): The slice length of the waveform.
        """
        super().__init__()

        # Parameters
        self.input_channels = input_channels
        self.hidden_dim = params.hidden_dim
        self.num_encoders = params.num_encoders
        self.num_decoders = params.num_decoders
        self.num_heads = params.num_heads
        self.vocabulary_size = vocabulary_size
        self.slice_length = slice_len
        self.activation = nn.LeakyReLU()

        # Initialize positional encoding
        self.positional_embedding = SinusoidalPositionEmbeddings(params.hidden_dim)

        # Initialize transformer encoder and decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.hidden_dim,
            nhead=params.num_heads,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, params.num_encoders)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=params.hidden_dim,
            nhead=params.num_heads,
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, params.num_decoders)

        # Input and output projections
        self.input_projection = nn.Linear(self.input_channels, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.vocabulary_size)
        self.decoder_query = nn.Embedding(self.slice_length, self.hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            z (torch.Tensor): Tensor size BS x C x InLen.

        Returns:
            torch.Tensor: Tensor size BS x SliceLen x VocabSize.
        """
        z = z.flatten(start_dim=1, end_dim=-2)
        proj_input = self.input_projection(z.transpose(1, 2).contiguous())
        proj_input = apply_pos_encoding(proj_input, self.positional_embedding)
        memory = self.encoder(proj_input)

        # Create query tensor
        query = self.decoder_query(
            torch.arange(self.slice_length).to(z.device).unsqueeze(0)
        ).repeat(z.shape[0], 1, 1)
        decoder_output = self.decoder(query, memory)
        return (
            self.output_projection(self.activation(decoder_output))
            .transpose(1, 2)
            .contiguous()
        )
