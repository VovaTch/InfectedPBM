import torch
import pytest

from loss.components.lm_cross_entropy import DecoderCrossEntropy
from common import registry


@pytest.fixture
def decoder_ce_loss() -> DecoderCrossEntropy:
    return DecoderCrossEntropy("decoder_ce", 1.0, registry.get_loss_module("ce"))


def test_decoder_cross_entropy_loss(decoder_ce_loss):
    """
    Test function for decoder cross entropy loss.

    Args:
        decoder_ce_loss: The decoder cross entropy loss function.

    Returns:
        None
    """
    estimation = {
        "logits": torch.tensor([[[0.2, 0.8], [0.6, 0.4]], [[0.1, 0.9], [0.3, 0.7]]])
    }
    target = {"latent indices": torch.tensor([[1, 0], [1, 1]])}
    result = decoder_ce_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.6178134083747864))
