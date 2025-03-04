import torch

from models.models.discriminator.attn_body import PatchAttentionDiscriminator
from models.models.discriminator.mel_spec_disc import MelSpecDiscriminator
from models.models.discriminator.stft_disc import StftDiscriminator
from models.models.discriminator.waveform_disc import WaveformDiscriminator
from models.models.discriminator.ensemble import EnsembleDiscriminator


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


def test_mel_spec_discriminator_no_res_forward(
    mel_spec_discriminator: MelSpecDiscriminator,
) -> None:
    input_tensor = torch.randn(3, 1, 32768)
    output = mel_spec_discriminator.forward(input_tensor)
    assert output["logits"].shape == (3, 20, 1)


def test_waveform_discriminator_no_res_forward(
    waveform_discriminator: WaveformDiscriminator,
) -> None:
    input_tensor = torch.randn(3, 1, 32768)
    output = waveform_discriminator.forward(input_tensor)
    assert output["logits"].shape == (3, 512, 1)


def test_ensemble_discriminator(ensemble_discriminator: EnsembleDiscriminator) -> None:
    input_tensor = torch.randn(3, 1, 32768)
    output = ensemble_discriminator.forward(input_tensor)
    assert output["logits"].shape == (3, 892, 1)
