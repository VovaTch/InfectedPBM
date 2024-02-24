import pytest
import torch
from omegaconf import DictConfig

from loaders.datasets import MP3TokenizedIndicesDataset
from loaders.datasets.music import MP3SliceDataset
from models.base import Tokenizer
from models.multi_level_vqvae.multi_level_vqvae import (
    MultiLvlVQVariationalAutoEncoder,
    RippleVQVariationalAutoEncoder,
)
from utils.containers import MusicDatasetParameters


@pytest.fixture
def dataset_params(cfg: DictConfig) -> MusicDatasetParameters:
    # Define the dataset parameters with dummy values
    return MusicDatasetParameters.from_cfg(cfg)


@pytest.fixture
def slice_dataset(cfg: DictConfig) -> MP3SliceDataset:
    # Define the slice dataset with dummy values
    return MP3SliceDataset.from_cfg(cfg)


@pytest.fixture
def tokenizer(cfg: DictConfig) -> Tokenizer:
    # Define the tokenizer with dummy values
    return MultiLvlVQVariationalAutoEncoder.from_cfg(cfg)


@pytest.fixture
def codebook_size(cfg: DictConfig) -> int:
    # Define the codebook size with a dummy value
    return cfg.model.vocabulary_size


@pytest.fixture
def index_series_length() -> int:
    # Define the index series length with a dummy value
    return 1024


@pytest.fixture
def buffer_process_batch_size() -> int:
    # Define the buffer process batch size with a dummy value
    return 32


@pytest.fixture
def epoch_size() -> int:
    # Define the epoch size with a dummy value
    return 100000


def test_MP3TokenizedIndicesDataset_init(
    dataset_params: MusicDatasetParameters,
    slice_dataset: MP3SliceDataset,
    tokenizer: Tokenizer,
    codebook_size: int,
    index_series_length: int,
    buffer_process_batch_size: int,
    epoch_size: int,
) -> None:
    # Test the initialization of MP3TokenizedIndicesDataset
    dataset = MP3TokenizedIndicesDataset(
        dataset_params=dataset_params,
        codebook_size=codebook_size,
        index_series_length=index_series_length,
        slice_dataset=slice_dataset,
        tokenizer=tokenizer,
        buffer_process_batch_size=buffer_process_batch_size,
        epoch_size=epoch_size,
    )
    assert isinstance(dataset, MP3TokenizedIndicesDataset)
    assert dataset.tokenized_data.shape == (6,)  # Check the shape of tokenized_data


def test_MP3TokenizedIndicesDataset_len(
    dataset_params: MusicDatasetParameters,
    slice_dataset: MP3SliceDataset,
    tokenizer: Tokenizer,
    codebook_size: int,
    index_series_length: int,
    buffer_process_batch_size: int,
    epoch_size: int,
):
    # Test the __len__ method of MP3TokenizedIndicesDataset
    dataset = MP3TokenizedIndicesDataset(
        dataset_params=dataset_params,
        codebook_size=codebook_size,
        index_series_length=index_series_length,
        slice_dataset=slice_dataset,
        tokenizer=tokenizer,
        buffer_process_batch_size=buffer_process_batch_size,
        epoch_size=epoch_size,
    )
    assert len(dataset) == 5  # Check the length of the dataset


def test_MP3TokenizedIndicesDataset_getitem(
    dataset_params: MusicDatasetParameters,
    slice_dataset: MP3SliceDataset,
    tokenizer: Tokenizer,
    codebook_size: int,
    index_series_length: int,
    buffer_process_batch_size: int,
    epoch_size: int,
):
    # Test the __getitem__ method of MP3TokenizedIndicesDataset
    dataset = MP3TokenizedIndicesDataset(
        dataset_params=dataset_params,
        codebook_size=codebook_size,
        index_series_length=index_series_length,
        slice_dataset=slice_dataset,
        tokenizer=tokenizer,
        buffer_process_batch_size=buffer_process_batch_size,
        epoch_size=epoch_size,
    )
    item = dataset[0]
    assert "indices" in item
    assert "target" in item
    assert item["indices"].shape == (1024,)
    assert item["target"].shape == (1024,)
