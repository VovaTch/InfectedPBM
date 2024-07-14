import pytest
import torch
import torch.nn as nn

from models.multi_level_vqvae.blocks import VQ1D
from models.multi_level_vqvae.multi_level_vqvae.ml_vqvae import (
    MultiLvlVQVariationalAutoEncoder,
)
from models.pipeline import AutoRegressivePipeline
from models.autoregressor_module import AutoRegressorModule
from models.base import Tokenizer


NUM_CLASSES_TESTING = 128


@pytest.fixture
def multi_lvl_vqvae() -> MultiLvlVQVariationalAutoEncoder:
    # Create a dummy instance of MultiLvlVQVariationalAutoEncoder for testing
    input_channels = 3
    encoder = nn.Conv1d(3, 64, kernel_size=3, padding=1)
    decoder = nn.Conv1d(64, 3, kernel_size=3, padding=1)
    vq_module = VQ1D(64, 128)
    return MultiLvlVQVariationalAutoEncoder(input_channels, encoder, decoder, vq_module)


@pytest.fixture
def mockup_regressor() -> AutoRegressorModule:
    class MockupRegressor:
        def __init__(self):
            self.model = MockupModel()

    class MockupModel:
        def get_last_logits(self, index_sequence: torch.Tensor) -> torch.Tensor:
            return torch.randn(
                index_sequence.shape[0], NUM_CLASSES_TESTING, index_sequence.shape[2]
            )

    return MockupRegressor()  # type: ignore


@pytest.fixture
def pipeline(
    mockup_regressor: AutoRegressorModule,
    multi_lvl_vqvae: MultiLvlVQVariationalAutoEncoder,
) -> AutoRegressivePipeline:
    return AutoRegressivePipeline(multi_lvl_vqvae, mockup_regressor)


def test_predict_next_token_no_top_k(pipeline: AutoRegressivePipeline) -> None:
    # Test case 1: Predict next token with default temperature and top_k
    index_sequence = torch.randint(0, NUM_CLASSES_TESTING, (64, 256, 16))
    predicted_tokens = pipeline.predict_next_token(index_sequence)
    assert predicted_tokens.shape == (64, 1, 16)


def test_predict_next_token_top_k(pipeline: AutoRegressivePipeline) -> None:
    # Test case 1: Predict next token with default temperature and top_k
    index_sequence = torch.randint(0, NUM_CLASSES_TESTING, (64, 256, 16))
    predicted_tokens = pipeline.predict_next_token(index_sequence, top_k=5)
    assert predicted_tokens.shape == (64, 1, 16)
