import torch

from models.models.multi_level_vqvae.encoder.stft_conv import EncoderConv2D


def test_stft_encoder_forward(stft_encoder: EncoderConv2D) -> None:
    dummy_input = torch.randn((3, 1, 32768))
    dummy_output = stft_encoder.forward(dummy_input)
    assert dummy_output.shape == (3, 32, 256)
