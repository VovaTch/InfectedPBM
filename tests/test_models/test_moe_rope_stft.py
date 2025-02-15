import torch
from models.models.moe_attention_layers.attention import FlashRotarySelfAttention
from models.models.moe_attention_layers.moe_transformer_block import (
    MoETransformerDecoderBlock,
)
from models.models.multi_level_vqvae.decoder.attention_stft import AttentionStftDecoder


def test_moe_rope_stft_forward(moe_rope_stft_decoder: AttentionStftDecoder) -> None:
    dummy_input = torch.randn(3, 128, 96)  # BS x C x L
    dummy_output = moe_rope_stft_decoder.forward(dummy_input)
    assert dummy_output.shape == (3, 1, 96 * 256)  # Latent length * hop length


def test_moe_rope_stft_forward_cuda(
    moe_rope_stft_decoder: AttentionStftDecoder,
) -> None:
    moe_rope_stft_decoder = moe_rope_stft_decoder.to("cuda")
    dummy_input = torch.randn(3, 128, 96).cuda()  # BS x C x L
    dummy_output = moe_rope_stft_decoder.forward(dummy_input)
    assert dummy_output.shape == (3, 1, 96 * 256)  # Latent length * hop length
