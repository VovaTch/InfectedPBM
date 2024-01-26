import torch
import pytest

from loss.components.codebook import CommitLoss, AlignLoss
from common import registry


@pytest.fixture
def commit_loss() -> CommitLoss:
    return CommitLoss("commit", 1.0, registry.get_loss_module("mse"))


@pytest.fixture
def align_loss() -> AlignLoss:
    return AlignLoss("align", 1.0, registry.get_loss_module("mse"))


def test_commit_loss(commit_loss):
    """
    Test function for the commit_loss function.

    Args:
        commit_loss: The commit_loss function to be tested.
    """
    estimation = {"emb": torch.tensor([[0.2, 0.8], [0.6, 0.4]])}
    target = {"z_e": torch.tensor([[0.1, 0.9], [0.3, 0.7]])}
    result = commit_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.05000000074505806))


def test_align_loss(align_loss):
    """
    Test the align_loss function.

    Args:
        align_loss: The align_loss function to be tested.
    """
    estimation = {"emb": torch.tensor([[0.2, 0.8], [0.6, 0.4]])}
    target = {"z_e": torch.tensor([[0.1, 0.9], [0.3, 0.7]])}
    result = align_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.05000000074505806))
