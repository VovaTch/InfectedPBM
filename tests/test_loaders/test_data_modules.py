import pytest
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import lightning as L

from loaders.data_modules import MusicDataModule
from utils.containers import LearningParameters, MusicDatasetParameters
from loaders.datasets import MP3SliceDataset


@pytest.fixture
def dataset_params(cfg: DictConfig) -> MusicDatasetParameters:
    return MusicDatasetParameters.from_cfg(cfg)


@pytest.fixture
def dataset(cfg: DictConfig) -> MP3SliceDataset:
    return MP3SliceDataset.from_cfg(cfg)


@pytest.fixture
def learning_params(cfg: DictConfig) -> LearningParameters:
    return LearningParameters.from_cfg(cfg)


@pytest.fixture
def music_data_module(
    learning_params: LearningParameters, dataset: MP3SliceDataset
) -> MusicDataModule:
    data_module = MusicDataModule(learning_params, dataset)
    data_module.setup("train")
    return data_module


def test_setup(
    music_data_module: MusicDataModule,
    learning_params: LearningParameters,
    dataset: MP3SliceDataset,
) -> None:
    music_data_module.setup("stage")
    assert len(music_data_module.train_dataset) == int(
        (1 - learning_params.val_split - learning_params.test_split) * len(dataset)
    )
    assert len(music_data_module.val_dataset) == len(dataset) - len(
        music_data_module.train_dataset
    ) - len(music_data_module.test_dataset)


def test_train_dataloader(
    music_data_module: MusicDataModule, learning_params: LearningParameters
) -> None:
    train_dataloader = music_data_module.train_dataloader()
    assert isinstance(train_dataloader, DataLoader)
    assert train_dataloader.batch_size == learning_params.batch_size
    assert train_dataloader.num_workers == learning_params.num_workers


def test_val_dataloader(
    music_data_module: MusicDataModule, learning_params: LearningParameters
) -> None:
    """
    Test the val_dataloader method of the MusicDataModule class.

    Args:
        music_data_module (MusicDataModule): An instance of the MusicDataModule class.
        learning_params (LearningParameters): An instance of the LearningParameters class.
    """
    val_dataloader = music_data_module.val_dataloader()
    assert isinstance(val_dataloader, DataLoader)
    assert val_dataloader.batch_size == learning_params.batch_size
    assert val_dataloader.num_workers == learning_params.num_workers


def test_test_dataloader(
    music_data_module: MusicDataModule, learning_params: LearningParameters
) -> None:
    """
    Test the test_dataloader method of the MusicDataModule class.

    Args:
        music_data_module (MusicDataModule): An instance of the MusicDataModule class.
        learning_params (LearningParameters): An instance of the LearningParameters class.
    """
    test_dataloader = music_data_module.test_dataloader()
    assert isinstance(test_dataloader, DataLoader)
    assert test_dataloader.batch_size == learning_params.batch_size
    assert test_dataloader.num_workers == learning_params.num_workers


def test_from_cfg(
    learning_params: MusicDataModule, dataset: MP3SliceDataset, cfg: DictConfig
) -> None:
    """
    Test the 'from_cfg' method of the MusicDataModule class.

    Args:
        learning_params (MusicDataModule): The expected learning parameters.
        dataset (MP3SliceDataset): The expected dataset.
    """

    music_data_module = MusicDataModule.from_cfg(cfg)  # type: ignore
    assert isinstance(music_data_module, MusicDataModule)
    assert music_data_module.learning_params == learning_params
    assert type(music_data_module.dataset) == type(dataset)
