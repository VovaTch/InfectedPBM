import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from utils.learning import LearningParameters


class SplitDatasetModule(L.LightningDataModule):
    """
    Simple data module to be used with standard datasets
    """

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, learning_params: LearningParameters, dataset: Dataset) -> None:
        """
        Initializer method

        Args:
            learning_params (LearningParameters): Learning parameter object
            dataset (MusicDataset): Dataset object
        """
        super().__init__()
        self._learning_params = learning_params
        self._dataset = dataset

    def setup(self, stage: str) -> None:
        """
        Lightning module setup method

        Args:
            stage (str): Unused in this implementation
        """

        if self._learning_params.test_split + self._learning_params.val_split > 1:
            raise ValueError(
                "Sum of test and validation split must be less than or equal to 1"
            )
        if self._learning_params.test_split < 0 or self._learning_params.val_split < 0:
            raise ValueError("Test and validation split must not be negative.")

        # Split into training and validation
        training_len = int(
            (1 - self._learning_params.val_split - self._learning_params.test_split)
            * len(self._dataset)  # type: ignore
        )
        test_len = int(training_len * self._learning_params.test_split)
        val_len = len(self._dataset) - training_len - test_len  # type: ignore

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self._dataset, lengths=(training_len, val_len, test_len)
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Lightning module train_dataloader method

        Returns:
            TRAIN_DATALOADERS: training dataloader object
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self._learning_params.batch_size,
            shuffle=True,
            num_workers=self._learning_params.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Lightning module eval_dataloader method

        Returns:
            EVAL_DATALOADERS: eval dataloader object
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self._learning_params.batch_size,
            shuffle=False,
            num_workers=self._learning_params.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Lightning module test_dataloader method

        Returns:
            EVAL_DATALOADERS: test dataloader object
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self._learning_params.batch_size,
            shuffle=False,
            num_workers=self._learning_params.num_workers,
        )
