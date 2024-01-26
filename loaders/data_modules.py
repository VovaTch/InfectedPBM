from typing import Protocol, Self, TypeVar

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from omegaconf import DictConfig
from torch.utils.data import random_split, DataLoader, Dataset

from common import registry
from loaders.datasets import MusicDataset
from utils.containers import LearningParameters

_T = TypeVar("_T")


class BaseDataModule(Protocol):
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def setup(self, stage: str) -> None:
        """
        Lightning module setup method

        Args:
            stage (str): Unused in this implementation
        """
        ...

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Lightning module train_dataloader method

        Returns:
            TRAIN_DATALOADERS: training dataloader object
        """
        ...

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Lightning module eval_dataloader method

        Returns:
            EVAL_DATALOADERS: eval dataloader object
        """
        ...

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Lightning module test_dataloader method

        Returns:
            EVAL_DATALOADERS: test dataloader object
        """
        ...

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Utility method to parse learning parameters from a configuration dictionary

        Args:
            cfg (DictConfig): configuration dictionary

        Returns:
            LearningParameters: Learning parameters object
        """
        ...


@registry.register_data_module("music_data_module")
class MusicDataModule(L.LightningDataModule):
    """
    Simple data module to be used with standard datasets
    """

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(
        self, learning_params: LearningParameters, dataset: MusicDataset
    ) -> None:
        """
        Initializer method

        Args:
            learning_params (LearningParameters): Learning parameter object
            dataset (MusicDataset): Dataset object
        """
        super().__init__()
        self.learning_params = learning_params
        self.dataset = dataset

    def setup(self, stage: str) -> None:
        """
        Lightning module setup method

        Args:
            stage (str): Unused in this implementation
        """

        if self.learning_params.test_split + self.learning_params.val_split > 1:
            raise ValueError(
                "Sum of test and validation split must be less than or equal to 1"
            )
        if self.learning_params.test_split < 0 or self.learning_params.val_split < 0:
            raise ValueError("Test and validation split must not be negative.")

        # Split into training and validation
        training_len = int(
            (1 - self.learning_params.val_split - self.learning_params.test_split)
            * len(self.dataset)
        )
        test_len = int(training_len * self.learning_params.test_split)
        val_len = len(self.dataset) - training_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, lengths=(training_len, val_len, test_len)  # type: ignore
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Lightning module train_dataloader method

        Returns:
            TRAIN_DATALOADERS: training dataloader object
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=True,
            num_workers=self.learning_params.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Lightning module eval_dataloader method

        Returns:
            EVAL_DATALOADERS: eval dataloader object
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Lightning module test_dataloader method

        Returns:
            EVAL_DATALOADERS: test dataloader object
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Utility method to parse learning parameters from a configuration dictionary

        Args:
            cfg (DictConfig): configuration dictionary

        Returns:
            LearningParameters: Learning parameters object
        """
        learning_params = LearningParameters.from_cfg(cfg)
        dataset = registry.get_dataset(cfg.dataset.dataset_type).from_cfg(cfg)
        return cls(learning_params, dataset)
