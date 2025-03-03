import os
import pytest

from loaders.datasets.latent import LatentSliceDataset
from loaders.datasets.music import MP3SliceDataset
from loaders.datasets.quantized import QuantizedUint8MusicDataset
from models.models.discriminator.attn_body import PatchAttentionDiscriminator
from models.models.discriminator.mlp_head import MLP
from models.models.discriminator.stft_disc import StftDiscriminator
from models.models.multi_level_vqvae.blocks.vq1d import VQ1D
from models.models.multi_level_vqvae.decoder.attention_stft import AttentionStftDecoder
from models.models.multi_level_vqvae.decoder.moe_stft import (
    MixtureOfExpertsRotaryStftDecoder,
)
from models.models.multi_level_vqvae.encoder.conv import Encoder1D
from models.models.multi_level_vqvae.encoder.stft_conv import EncoderConv2D
from models.models.multi_level_vqvae.ml_vqvae import MultiLvlVQVariationalAutoEncoder


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
def encoder1d() -> Encoder1D:
    return Encoder1D(
        input_channels=1, dim_change_list=[4, 4, 4, 2], channel_list=[2, 4, 8, 16, 128]
    )


@pytest.fixture
def vq_module() -> VQ1D:
    return VQ1D(
        token_dim=128,
        num_tokens=512,
        num_rq_steps=4,
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
def vqvae_basic(
    encoder1d: Encoder1D, attention_stft_decoder: AttentionStftDecoder, vq_module: VQ1D
) -> MultiLvlVQVariationalAutoEncoder:
    return MultiLvlVQVariationalAutoEncoder(
        input_channels=1,
        encoder=encoder1d,
        decoder=attention_stft_decoder,
        vq_module=vq_module,
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


@pytest.fixture
def lvl1_latent_dataset(
    vqvae_basic: MultiLvlVQVariationalAutoEncoder,
) -> LatentSliceDataset:
    return LatentSliceDataset(
        data_path=os.path.join("tests", "data_slices"),
        slice_level=1,
        slices_per_sample=16,
        tokenizer=vqvae_basic,
        device="cpu",
    )


@pytest.fixture
def mlp_class_head() -> MLP:
    return MLP(
        input_dim=128,
        hidden_dim=256,
        num_layers=3,
        output_dim=2,
        dropout=0.1,
    )


@pytest.fixture
def attn_discriminator(
    mlp_class_head: MLP, encoder1d: Encoder1D
) -> PatchAttentionDiscriminator:
    return PatchAttentionDiscriminator(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        patch_size=2,
        feature_extractor=encoder1d,
        class_head=mlp_class_head,
        dropout=0.1,
    )


@pytest.fixture
def stft_discriminator() -> StftDiscriminator:
    return StftDiscriminator(
        channel_list=[2, 4, 8, 16, 32],
        n_fft=256,
        hop_length=64,
        win_length=256,
        stride=2,
        kernel_size=7,
    )


@pytest.fixture
def stft_encoder() -> EncoderConv2D:
    return EncoderConv2D(
        channel_list=[2, 4, 8, 16, 32],
        dim_change_list=[2, 2, 2, 2],
        n_fft=256,
        hop_length=64,
        win_length=256,
        kernel_size=3,
        num_res_block_conv=3,
    )
