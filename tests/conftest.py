import os
import pytest

from loaders.datasets.music import MP3SliceDataset
from loaders.datasets.quantized import QuantizedUint8MusicDataset
from models.models.multi_level_vqvae.decoder.attention_stft import AttentionStftDecoder
from models.models.multi_level_vqvae.decoder.moe_stft import (
    MixtureOfExpertsRotaryStftDecoder,
)


@pytest.fixture
def mp3_dataset() -> MP3SliceDataset:
    return MP3SliceDataset(
        data_path="tests/data", sample_rate=44100, slice_length=1024, device="cpu"
    )


@pytest.fixture
def music_quant_test_dataset() -> QuantizedUint8MusicDataset:
    return QuantizedUint8MusicDataset(
        data_path=os.path.join("tests", "data"), slice_length=1024, sample_rate=44100
    )


@pytest.fixture
def attention_stft_decoder() -> AttentionStftDecoder:
    return AttentionStftDecoder(
        hidden_dim=256,
        input_dim=128,
        num_layers=3,
        num_heads=4,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        dropout=0.1,
        padding="same",
    )


@pytest.fixture
def moe_rope_stft_decoder() -> MixtureOfExpertsRotaryStftDecoder:
    return MixtureOfExpertsRotaryStftDecoder(
        input_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=3,
        num_experts=4,
        top_k_gating=2,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        max_seq_len=32768,
        ff_hidden_dim=2048,
        norm_type="layernorm",
        dropout=0.1,
        padding="same",
        use_causal=True,
    )
