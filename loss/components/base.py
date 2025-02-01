from abc import ABC, abstractmethod

import torch


class LossComponent(ABC):
    """
    Basic loss component base class, using the __call__ method to compute the loss.

    Attributes:
        name (str): The name of the loss component.
        differentiable (bool): Whether the loss component is differentiable.
        weight (float): The weight of the loss component.
    """

    name: str
    differentiable: bool
    weight: float

    @abstractmethod
    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the loss.

        Args:
            pred (dict[str, torch.Tensor]): A dictionary containing the predicted values.
            target (dict[str, torch.Tensor]): A dictionary containing the target values.

        Returns:
            torch.Tensor: The computed loss as a tensor.
        """
        ...
