import lightning as L
from torch.utils.data import DataLoader, Dataset

from utils.learning import LearningParameters


class SeparatedSetModule(L.LightningDataModule):
    """
    LightningDataModule subclass for managing separated datasets (train, validation, and test) in a PyTorch Lightning project.

    Args:
        learning_params (LearningParameters): A data class or dictionary containing various learning parameters.
        train_dataset (Dataset): The dataset used for training.
        val_dataset (Dataset): The dataset used for validation.
        test_dataset (Dataset | None, optional): The dataset used for testing. If not provided,
        the validation dataset is used for testing.

    Attributes:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The testing dataset.

    Note:
        The `SeparatedSetModule` class is designed to manage and provide data loaders for the training,
        validation, and testing phases of a deep learning project.
    """

    def __init__(
        self,
        learning_params: LearningParameters,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset | None = None,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset if test_dataset is not None else val_dataset
        self.learning_params = learning_params

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: A DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=True,
            num_workers=self.learning_params.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: A DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the testing dataset.

        Returns:
            DataLoader: A DataLoader for the testing dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )
