from typing import Callable, TYPE_CHECKING, TypeVar
from dataclasses import dataclass, field

from omegaconf import DictConfig
import torch
import torch.nn as nn
import lightning as L

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator
    from loss.component_base import LossComponent
    from models.mel_spec_converters import MelSpecConverter
    from loaders.datasets import MusicDataset
    from loaders.data_modules import BaseDataModule
    from models.base import BaseLightningModule


LossComponentFactory = Callable[[str, DictConfig], "LossComponent"]
TransformFunction = Callable[[torch.Tensor], torch.Tensor]

T = TypeVar("T")


@dataclass
class Registry:
    """
    A class that serves as a registry container for various components in a software system.

    The Registry class allows registering and accessing different components such as models, activation functions,
    lightning modules, datasets, data modules, mel spec converters, loss components, loss modules, loss aggregators,
    schedulers, optimizers, and transform functions.

    Each component type has methods for registering, getting, and deleting components by name.

    Example usage:

    ```
    registry = Registry()

    # Register a model
    @registry.register_model("my_model")
    class MyModel(nn.Module):
        ...

    # Get a model
    model = registry.get_model("my_model")

    # Delete a model
    registry.delete_model("my_model")
    ```

    """

    models: dict[str, type[nn.Module]] = field(default_factory=lambda: {})
    activation_functions: dict[str, nn.Module] = field(default_factory=lambda: {})
    lightning_modules: dict[str, type["BaseLightningModule"]] = field(
        default_factory=lambda: {}
    )
    datasets: dict[str, type["MusicDataset"]] = field(default_factory=lambda: {})
    data_modules: dict[str, type["BaseDataModule"]] = field(default_factory=lambda: {})
    mel_spec_converters: dict[str, type["MelSpecConverter"]] = field(
        default_factory=lambda: {}
    )
    loss_components: dict[str, type["LossComponent"]] = field(
        default_factory=lambda: {}
    )
    loss_modules: dict[str, nn.Module] = field(default_factory=lambda: {})
    loss_aggregators: dict[str, type["LossAggregator"]] = field(
        default_factory=lambda: {}
    )
    schedulers: dict[str, type[torch.optim.lr_scheduler._LRScheduler]] = field(
        default_factory=lambda: {}
    )
    optimizers: dict[str, type[torch.optim.Optimizer]] = field(
        default_factory=lambda: {}
    )
    transforms_functions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = field(
        default_factory=lambda: {}
    )

    # MODELS
    def register_model(self, name: str) -> Callable[[type[T]], type[T]]:
        """
        Register a model with the given name.

        Args:
            name (str): The name of the model.

        Returns:
            Callable[[type[T]], type[T]]: A decorator function that registers the model.

        Raises:
            KeyError: If a model with the same name is already registered.

        """

        def wrapper(model: type[T]) -> type[T]:
            if name in self.models:
                raise KeyError(f"Model {name} is already registered")
            self.models[name] = model  # type: ignore
            return model

        return wrapper

    def get_model(self, name: str) -> type[nn.Module]:
        if name not in self.models:
            raise KeyError(
                f"Model {name} is not registered, available names: {list(self.models.keys())}"
            )
        return self.models[name]

    def delete_model(self, name: str) -> None:
        if name not in self.models:
            raise KeyError(
                f"Model {name} is not registered, available names: {list(self.models.keys())}"
            )
        del self.models[name]

    # ACTIVATION FUNCTIONS
    def register_activation_function(
        self, name: str
    ) -> Callable[[nn.Module], nn.Module]:
        def decorator(activation_function: nn.Module) -> nn.Module:
            if name in self.activation_functions:
                raise KeyError(f"Activation function {name} is already registered")
            self.activation_functions[name] = activation_function
            return activation_function

        return decorator

    def get_activation_function(self, name: str) -> nn.Module:
        if name not in self.activation_functions:
            raise KeyError(
                f"Activation function {name} is not registered, available names: {list(self.activation_functions.keys())}"
            )
        return self.activation_functions[name]

    def delete_activation_function(self, name: str) -> None:
        if name not in self.activation_functions:
            raise KeyError(
                f"Activation function {name} is not registered, available names: {list(self.activation_functions.keys())}"
            )
        del self.activation_functions[name]

    # LIGHTNING MODULES
    def register_lightning_module(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(lightning_module: type[T]) -> type[T]:
            if name in self.lightning_modules:
                raise KeyError(f"Lightning module {name} is already registered")
            self.lightning_modules[name] = lightning_module  # type: ignore
            return lightning_module

        return decorator

    def get_lightning_module(self, name: str) -> type["BaseLightningModule"]:
        if name not in self.lightning_modules:
            raise KeyError(
                f"Lightning module {name} is not registered, available names: {list(self.lightning_modules.keys())}"
            )
        return self.lightning_modules[name]

    def delete_lightning_module(self, name: str) -> None:
        if name not in self.lightning_modules:
            raise KeyError(
                f"Lightning module {name} is not registered, available names: {list(self.lightning_modules.keys())}"
            )
        del self.lightning_modules[name]

    # MEL SPEC CONVERTERS
    def register_mel_spec_converter(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(
            mel_spec_converter: type[T],
        ) -> type[T]:
            if name in self.mel_spec_converters:
                raise KeyError(f"MelSpecConverter {name} is already registered")
            self.mel_spec_converters[name] = mel_spec_converter  # type: ignore
            return mel_spec_converter

        return decorator

    def get_mel_spec_converter(self, name: str) -> type["MelSpecConverter"]:
        if name not in self.mel_spec_converters:
            raise KeyError(
                f"MelSpecConverter {name} is not registered, available names: {list(self.mel_spec_converters.keys())}"
            )
        return self.mel_spec_converters[name]

    def delete_mel_spec_converter(self, name: str) -> None:
        if name not in self.mel_spec_converters:
            raise KeyError(
                f"MelSpecConverter {name} is not registered, available names: {list(self.mel_spec_converters.keys())}"
            )
        del self.mel_spec_converters[name]

    # LOSS COMPONENTS
    def register_loss_component(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(loss_component: type[T]) -> type[T]:
            if name in self.loss_components:
                raise KeyError(f"Loss component {name} is already registered")
            self.loss_components[name] = loss_component  # type: ignore
            return loss_component

        return decorator

    def get_loss_component(self, name: str) -> type["LossComponent"]:
        if name not in self.loss_components:
            raise KeyError(
                f"Loss component {name} is not registered, available names: {list(self.loss_components.keys())}"
            )
        return self.loss_components[name]

    def delete_loss_component(self, name: str) -> None:
        if name not in self.loss_components:
            raise KeyError(
                f"Loss component {name} is not registered, available names: {list(self.loss_components.keys())}"
            )
        del self.loss_components[name]

    # LOSS MODULES
    def register_loss_module(self, name: str) -> Callable[[nn.Module], nn.Module]:
        def decorator(loss_module: nn.Module) -> nn.Module:
            if name in self.loss_modules:
                raise KeyError(f"Loss module {name} is already registered")
            self.loss_modules[name] = loss_module
            return loss_module

        return decorator

    def get_loss_module(self, name: str) -> nn.Module:
        if name not in self.loss_modules:
            raise KeyError(
                f"Loss module {name} is not registered, available names: {list(self.loss_modules.keys())}"
            )
        return self.loss_modules[name]

    def delete_loss_module(self, name: str) -> None:
        if name not in self.loss_modules:
            raise KeyError(
                f"Loss module {name} is not registered, available names: {list(self.loss_modules.keys())}"
            )
        del self.loss_modules[name]

    # LOSS AGGREGATORS
    def register_loss_aggregator(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(
            loss_aggregator: type[T],
        ) -> type[T]:
            if name in self.loss_aggregators:
                raise KeyError(f"Loss aggregator {name} is already registered")
            self.loss_aggregators[name] = loss_aggregator  # type: ignore
            return loss_aggregator

        return decorator

    def get_loss_aggregator(self, name: str) -> type["LossAggregator"]:
        if name not in self.loss_aggregators:
            raise KeyError(
                f"Loss aggregator {name} is not registered, available names: {list(self.loss_aggregators.keys())}"
            )
        return self.loss_aggregators[name]

    def delete_loss_aggregator(self, name: str) -> None:
        if name not in self.loss_aggregators:
            raise KeyError(
                f"Loss aggregator {name} is not registered, available names: {list(self.loss_aggregators.keys())}"
            )
        del self.loss_aggregators[name]

    # SCHEDULERS
    def register_scheduler(self, name: str) -> Callable[
        [type[torch.optim.lr_scheduler._LRScheduler]],
        type[torch.optim.lr_scheduler._LRScheduler],
    ]:
        def decorator(
            scheduler: type[torch.optim.lr_scheduler._LRScheduler],
        ) -> type[torch.optim.lr_scheduler._LRScheduler]:
            if name in self.schedulers:
                raise KeyError(f"Scheduler {name} is already registered")
            self.schedulers[name] = scheduler  # tyoe: ignore
            return scheduler

        return decorator

    def get_scheduler(self, name: str) -> type[torch.optim.lr_scheduler._LRScheduler]:
        if name not in self.schedulers:
            raise KeyError(
                f"Scheduler {name} is not registered, available names: {list(self.schedulers.keys())}"
            )
        return self.schedulers[name]

    def delete_scheduler(self, name: str) -> None:
        """
        Deletes a registered scheduler by name.

        Args:
            name (str): The name of the scheduler to delete.

        Raises:
            KeyError: If the specified scheduler is not registered.
        """
        if name not in self.schedulers:
            raise KeyError(
                f"Scheduler {name} is not registered, available names: {list(self.schedulers.keys())}"
            )
        del self.schedulers[name]

    # OPTIMIZERS
    def register_optimizer(
        self, name: str
    ) -> Callable[[type[torch.optim.Optimizer]], type[torch.optim.Optimizer]]:
        def decorator(
            optimizer: type[torch.optim.Optimizer],
        ) -> type[torch.optim.Optimizer]:
            if name in self.optimizers:
                raise KeyError(f"Optimizer {name} is already registered")
            self.optimizers[name] = optimizer
            return optimizer

        return decorator

    def get_optimizer(self, name: str) -> type[torch.optim.Optimizer]:
        if name not in self.optimizers:
            raise KeyError(
                f"Optimizer {name} is not registered, available names: {list(self.optimizers.keys())}"
            )
        return self.optimizers[name]

    def delete_optimizer(self, name: str) -> None:
        if name not in self.optimizers:
            raise KeyError(
                f"Optimizer {name} is not registered, available names: {list(self.optimizers.keys())}"
            )
        del self.optimizers[name]

    # TRANSFORM FUNCTIONS
    def register_transform_function(
        self, name: str
    ) -> Callable[[TransformFunction], TransformFunction]:
        def decorator(transform_function: TransformFunction) -> TransformFunction:
            if name in self.transforms_functions:
                raise KeyError(f"Activation function {name} is already registered")
            self.transforms_functions[name] = transform_function
            return transform_function

        return decorator

    def get_transform_function(
        self, name: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if name not in self.transforms_functions:
            raise KeyError(
                f"Transformation function {name} is not registered, available names: {list(self.transforms_functions.keys())}"
            )
        return self.transforms_functions[name]

    def delete_transform_function(self, name: str) -> None:
        if name not in self.transforms_functions:
            raise KeyError(
                f"Transformation function {name} is not registered, available names: {list(self.transforms_functions.keys())}"
            )
        del self.transforms_functions[name]

    # DATA MODULES
    def register_data_module(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(
            data_module: type[T],
        ) -> type[T]:
            if name in self.data_modules:
                raise KeyError(f"Lightning data module {name} is already registered")
            self.data_modules[name] = data_module  # type: ignore
            return data_module

        return decorator

    def get_data_module(self, name: str) -> type["BaseDataModule"]:
        if name not in self.data_modules:
            raise KeyError(
                f"Lightning data module {name} is not registered, available names: {list(self.data_modules.keys())}"
            )
        return self.data_modules[name]

    def delete_data_module(self, name: str) -> None:
        if name not in self.data_modules:
            raise KeyError(
                f"Lightning data module {name} is not registered, available names: {list(self.data_modules.keys())}"
            )
        del self.data_modules[name]

    # MUSIC DATASETS
    def register_dataset(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(dataset: type[T]) -> type[T]:
            if name in self.datasets:
                raise KeyError(f"MusicDataset {name} is already registered")
            self.datasets[name] = dataset  # type: ignore
            return dataset

        return decorator

    def get_dataset(self, name: str) -> type["MusicDataset"]:
        if name not in self.datasets:
            raise KeyError(
                f"Dataset {name} is not registered, available names: {list(self.datasets.keys())}"
            )
        return self.datasets[name]

    def delete_dataset(self, name: str) -> None:
        if name not in self.datasets:
            raise KeyError(
                f"Dataset {name} is not registered, available names: {list(self.datasets.keys())}"
            )
        del self.datasets[name]


registry = Registry()
"""
Responsible for registering and retrieving all components of the pipeline, called via a string, can
be initialized through a configuration file.
"""
