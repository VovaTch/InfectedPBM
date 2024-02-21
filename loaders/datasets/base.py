from typing import Any, Protocol
from typing_extensions import Self

from omegaconf import DictConfig

from utils.containers import MusicDatasetParameters


class MusicDataset(Protocol):
    """
    Basic music dataset protocol
    """

    dataset_params: MusicDatasetParameters
    buffer: dict[str, Any] = {}  # Data buffer dictionary

    def __init__(self, dataset_params: MusicDatasetParameters) -> None:
        """
        Initializer method

        Args:
            dataset_params (MusicDatasetParameters): Dataset parameter object
        """
        ...

    def _dump_data(self, path: str) -> None:
        """
        Saves the data in a designated folder path

        Args:
            path (str): Saved data folder path
        """
        ...

    def _load_data(self, path: str) -> None:
        """
        Loads the data from a designated folder path

        Args:
            path (str): Loaded data folder path
        """
        ...

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Standard dataset object __getitem__ method

        Args:
            index (int): Index of the data-point

        Returns:
            dict[str, Any]: Dictionary item from the dataset, collected values with a
            collate_fn function from Pytorch
        """
        ...

    def __len__(self) -> int:
        """
        Dataset length getter method

        Returns:
            int: Dataset length
        """
        ...

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Utility method to parse dataset parameters from a configuration dictionary

        Args:
            cfg (DictConfig): configuration dictionary

        Returns:
            MusicDataset: dataset object
        """
        ...
