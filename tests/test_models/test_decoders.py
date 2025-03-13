import torch

from models.models.multi_level_vqvae.decoder.conv1d_stft import StftDecoder1D
from models.models.multi_level_vqvae.decoder.conv2d_stft import StftDecoder2D


def test_conv_stft_decoder_forward(conv_stft_decoder: StftDecoder1D) -> None:
    dummy_input = torch.randn(3, 64, 32)  # BS x C x L
    dummy_output = conv_stft_decoder.forward(dummy_input)
    assert dummy_output.shape == (3, 1, 32768)  # L * prod(dim_change) * hop_length


def test_conv_stft_decoder_2d_forward(conv_stft_decoder_2d: StftDecoder2D) -> None:
    dummy_input = torch.randn(3, 64, 32)  # BS x C x L x L
    dummy_output = conv_stft_decoder_2d.forward(dummy_input)
    assert dummy_output.shape == (3, 1, 32768)  # L * prod(dim_change) * hop_length
