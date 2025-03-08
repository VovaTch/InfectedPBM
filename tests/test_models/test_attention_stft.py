import torch

from models.models.multi_level_vqvae.decoder.attention_stft import AttentionStftDecoder


def test_attention_stft_decoder_forward(
    attention_stft_decoder: AttentionStftDecoder,
) -> None:
    dummy_input = torch.randn(3, 128, 96)  # BS x C x L
    dummy_output = attention_stft_decoder.forward(dummy_input)
    assert dummy_output.shape == (3, 1, 96 * 256)  # Latent length * hop length


def test_attention_stft_decoder_forward_cuda(
    attention_stft_decoder: AttentionStftDecoder,
) -> None:
    attention_stft_decoder = attention_stft_decoder.to("cuda")
    dummy_input = torch.randn(3, 128, 96).cuda()  # BS x C x L
    dummy_output = attention_stft_decoder.forward(dummy_input)
    assert dummy_output.shape == (3, 1, 96 * 256)  # Latent length * hop length
