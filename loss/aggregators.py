from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING
from typing_extensions import Self

from omegaconf import DictConfig
import torch

from common import registry

if TYPE_CHECKING:
    from .component_base import LossComponent


@dataclass
class LossOutput:
    """
    Loss output object that contains individual components named, and the total loss from the aggregator.

    Fields:
        *   total (Tensor): Total loss from the aggregator
        *   individuals (dict[str, Tensor]): Individual loss component values.
    """

    total: torch.Tensor
    individuals: dict[str, torch.Tensor]


class LossAggregator(Protocol):
    """
    Loss aggregator protocol, uses a math operation on component losses to compute a total loss. For example, weighted sum.
    """

    components: list["LossComponent"]

    def __call__(
        self,
        estimation: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> LossOutput:
        """Call method for loss aggregation

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation dictionary
            target (dict[str, torch.Tensor]): Target dictionary

        Returns:
            LossOutput: LossOutput object representing the total loss and the individual parts
        """
        ...

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create an instance of the class from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Returns:
            Self: An instance of the class.
        """
        ...


@registry.register_loss_aggregator("weighted_sum")
@dataclass
class WeightedSumAggregator:
    """
    Weighted sum loss component
    """

    components: list["LossComponent"]

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput:
        """
        Forward method to compute the weighted sum

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            LossOutput: Loss output object with total loss and individual losses
        """
        loss = LossOutput(torch.tensor(0.0), {})

        for component in self.components:
            ind_loss = component(estimation, target)
            if component.differentiable:
                loss.total = loss.total.to(ind_loss.device)
                loss.total += component.weight * ind_loss
            loss.individuals[component.name] = ind_loss

        return loss

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Utility method to parse loss parameters from a configuration dictionary

        Args:
            cfg (DictConfig): configuration dictionary

        Returns:
            WeightedSumAggregator: Weighted sum loss object
        """
        return cls(
            [
                registry.get_loss_component(loss_cfg["type"]).from_cfg(name, loss_cfg)
                for name, loss_cfg in cfg.loss.components.items()
            ]
        )
