import torch
import pytest

from loss.components.classification import BasicClassificationLoss, PercentCorrect
from common import registry


@pytest.fixture
def percent_correct_loss() -> PercentCorrect:
    return PercentCorrect("percent_correct", 1.0)


@pytest.fixture
def cls_loss() -> BasicClassificationLoss:
    return BasicClassificationLoss("basic_cls", 1.0, registry.get_loss_module("ce"))


def test_percent_correct(percent_correct_loss):
    estimation = {"pred_logits": torch.tensor([[0.2, 0.8], [0.6, 0.4]])}
    target = {"class": torch.tensor([1, 0])}
    result = percent_correct_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(1.0))

    estimation = {"pred_logits": torch.tensor([[0.1, 0.9], [0.3, 0.7]])}
    target = {"class": torch.tensor([1, 0])}
    result = percent_correct_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.5))

    estimation = {"pred_logits": torch.tensor([[0.4, 0.6], [0.8, 0.2]])}
    target = {"class": torch.tensor([1, 0])}
    result = percent_correct_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(1.0))

    estimation = {"pred_logits": torch.tensor([[0.9, 0.1], [0.7, 0.3]])}
    target = {"class": torch.tensor([0, 0])}
    result = percent_correct_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(1.0))


def test_basic_classification_loss(cls_loss):
    estimation = {"pred_logits": torch.tensor([[0.2, 0.8], [0.6, 0.4]])}
    target = {"class": torch.tensor([1, 0])}
    result = cls_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.51781344))

    estimation = {"pred_logits": torch.tensor([[0.1, 0.9], [0.3, 0.7]])}
    target = {"class": torch.tensor([1, 1])}
    result = cls_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.44205796))

    estimation = {"pred_logits": torch.tensor([[0.4, 0.6], [0.8, 0.2]])}
    target = {"class": torch.tensor([0, 0])}
    result = cls_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.6178134083747864))

    estimation = {"pred_logits": torch.tensor([[0.9, 0.1], [0.7, 0.3]])}
    target = {"class": torch.tensor([0, 1])}
    result = cls_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.6420579552650452))
