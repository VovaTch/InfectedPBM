import torch

from models.models.discriminator.attn_body import PatchAttentionDiscriminator
from models.models.discriminator.stft_disc import StftDiscriminator


def test_attn_discriminator_no_res_forward(
    attn_discriminator: PatchAttentionDiscriminator,
) -> None:
    input_tensor = torch.randn(3, 1, 512)
    output = attn_discriminator.forward(input_tensor)
    assert output["logits"].shape == (3, 2, 2)


def test_attn_discriminator_res_forward(
    attn_discriminator: PatchAttentionDiscriminator,
) -> None:
    input_tensor = torch.randn(3, 1, 512 + 128)
    output = attn_discriminator.forward(input_tensor)
    assert output["logits"].shape == (3, 3, 2)


def test_stft_discriminator_no_res_forward(
    stft_discriminator: StftDiscriminator,
) -> None:
    input_tensor = torch.randn(3, 1, 1024)
    output = stft_discriminator.forward(input_tensor)
    assert output["logits"].shape == (3, 81, 1)
