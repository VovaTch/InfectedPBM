import torch
import pytest
from omegaconf import OmegaConf, DictConfig
from models.mamba.multi_head import ChaosHydra


@pytest.fixture
def config():
    cfg = {
        "model": {
            "vocabulary_size": 100,
            "num_heads": 4,
            "mlp_head": {"hidden_dim": 256, "num_layers": 2, "activation": "relu"},
            "model_dim": 2048,
            "ssm_state_dim": 16,
            "conv_width": 3,
            "expansion": 2,
        },
    }
    return OmegaConf.create(cfg)


def test_chaos_hydra_forward(config: DictConfig) -> None:
    model = ChaosHydra.from_cfg(config).to("cuda")
    batch_size = 16
    seq_length = 32
    num_codebooks = 4
    input_tensor = torch.randint(
        0, config.model.vocabulary_size, (batch_size, seq_length, num_codebooks)
    ).to("cuda")
    output = model.forward(input_tensor)
    assert output["pred_logits"].shape == (
        batch_size,
        config.model.vocabulary_size + 2,
        seq_length,
        num_codebooks,
    )  # BS x Vocab_size+2 x L x num_CB


def test_chaos_hydra_get_last_logits(config: DictConfig) -> None:
    model = ChaosHydra.from_cfg(config).to("cuda")
    batch_size = 16
    seq_length = 32
    num_codebooks = 4
    input_tensor = torch.randint(
        0, config.model.vocabulary_size, (batch_size, seq_length, num_codebooks)
    ).to("cuda")
    last_logits = model.get_last_logits(input_tensor)
    assert last_logits.shape == (
        batch_size,
        config.model.vocabulary_size + 2,
        num_codebooks,
    )  # BS x Vocab_size+2 x num_CB
