from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Protocol

import torch


@dataclass
class LossOutput:
    """
    Represents the output of a loss calculation.

    Attributes:
        total (torch.Tensor): The total loss value.
        individual (dict[str, torch.Tensor]): A dictionary containing individual loss values for each component.
    """

    total: torch.Tensor
    individual: dict[str, torch.Tensor]


class LossAggregator(ABC):
    """
    Base class for loss aggregator.

    This class defines the protocol for a loss aggregator, which is responsible for aggregating
    the losses calculated by the model.
    """

    @abstractmethod
    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput:
        """
        Perform the forward pass of the loss aggregator.

        Args:
            pred (dict[str, torch.Tensor]): The predicted output of the model.
            target (dict[str, torch.Tensor]): The target output.

        Returns:
            LossOutput: The computed loss output.
        """
        ...

    @abstractmethod
    def recalculate_total_loss(self, loss: LossOutput) -> LossOutput:
        """
        Recalculates the total loss based on the provided LossOutput object.

        Args:
            loss (LossOutput): An object containing the current loss values.

        Returns:
            LossOutput: An object containing the recalculated total loss values.
        """
        ...


class LossComponent(Protocol):
    name: str
    differentiable: bool
    weight: float

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor: ...


class WeightedSumAggregator(LossAggregator):
    """
    Aggregator that computes the weighted sum of multiple loss components.

    Args:
        components (Iterable[LossComponent]): A collection of loss components.

    Returns:
        LossOutput: The aggregated loss output.

    Example:

    ```python
    aggregator = WeightedSumAggregator([component1, component2])
    loss_output = aggregator(pred, target)
    ```
    """

    def __init__(self, components: Iterable[LossComponent]) -> None:
        """
        Initializes the Aggregator object.

        Args:
            components (Iterable[LossComponent]): An iterable of LossComponent objects.
        """
        self.components = components

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput:
        """
        Calculates the aggregated loss based on the predictions and targets.

        Args:
            pred (dict[str, torch.Tensor]): A dictionary containing the predicted values.
            target (dict[str, torch.Tensor]): A dictionary containing the target values.

        Returns:
            LossOutput: An instance of the LossOutput class representing the aggregated loss.
        """
        loss = LossOutput(torch.tensor(0.0), {})

        for component in self.components:
            ind_loss = component(pred, target)
            if component.differentiable:
                loss.total = loss.total.to(ind_loss.device)
                loss.total = loss.total + component.weight * ind_loss
            loss.individual[component.name] = ind_loss

        return loss

    def recalculate_total_loss(self, loss: LossOutput) -> LossOutput:
        """
        Recalculates the total loss by summing the weighted individual losses of the differentiable components.

        Args:
            loss (LossOutput): An instance of LossOutput containing the total and individual losses.

        Returns:
            LossOutput: A new LossOutput instance with the recalculated total loss and the same individual losses.
        """
        recalculated_loss = LossOutput(torch.tensor(0.0).to(loss.total.device), {})
        for component in self.components:
            if component.differentiable:
                recalculated_loss.total = recalculated_loss.total + (
                    component.weight * loss.individual[component.name]
                )
            recalculated_loss.individual[component.name] = loss.individual[
                component.name
            ]

        return recalculated_loss
