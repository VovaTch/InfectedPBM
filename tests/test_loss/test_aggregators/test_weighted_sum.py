from dataclasses import dataclass
from numpy import diff
import torch
import torch.nn as nn
import pytest

from loss.aggregators import WeightedSumAggregator
from loss.component_base import LossComponent
from common import registry


@pytest.fixture
def loss_component() -> type[LossComponent]:
    @dataclass
    class MockLossComponent:
        name: str
        weight: float
        base_loss: nn.Module
        differentiable: bool = True

        def __call__(self, estimation, target) -> torch.Tensor:
            return torch.tensor(0.5)

    return MockLossComponent("mock_loss", 1.0, registry.get_loss_module("mse"))  # type: ignore


@pytest.fixture
def weighted_sum_aggregator(loss_component):
    return WeightedSumAggregator([loss_component])


def test_weighted_sum_aggregator(weighted_sum_aggregator):
    """
    Test the weighted sum aggregator function.

    Args:
        weighted_sum_aggregator: The weighted sum aggregator function to be tested.

    Returns:
        None
    """
    estimation = {"pred_logits": torch.tensor([[0.2, 0.8], [0.6, 0.4]])}
    target = {"class": torch.tensor([1, 0])}

    result = weighted_sum_aggregator(estimation, target)

    assert torch.isclose(result.total, torch.tensor(0.5))
    assert result.individuals == {"mock_loss": torch.tensor(0.5)}


def test_weighted_sum_aggregator_from_cfg(
    weighted_sum_aggregator: WeightedSumAggregator,
):
    """
    Test the `weighted_sum_aggregator_from_cfg` function.

    Args:
        weighted_sum_aggregator (WeightedSumAggregator): The weighted sum aggregator object to test.

    Raises:
        AssertionError: If any of the assertions fail.

    """
    assert len(weighted_sum_aggregator.components) == 1
    assert weighted_sum_aggregator.components[0].name == "mock_loss"
    assert weighted_sum_aggregator.components[0].weight == 1.0
    assert weighted_sum_aggregator.components[0].differentiable is True


def test_aggregator_forward(weighted_sum_aggregator: WeightedSumAggregator) -> None:
    """
    Test the forward method of the aggregator.

    Args:
        weighted_sum_aggregator (WeightedSumAggregator): The aggregator to be tested.
    """
    estimation = {"pred_logits": torch.tensor([[0.2, 0.8], [0.6, 0.4]])}
    target = {"class": torch.tensor([1, 0])}
    result = weighted_sum_aggregator(estimation, target)
    assert torch.isclose(result.total, torch.tensor(0.5))
    assert result.individuals == {"mock_loss": torch.tensor(0.5)}
