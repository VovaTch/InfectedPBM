import os
import torch
import pytest

from omegaconf import DictConfig

from utils.containers import MusicDatasetParameters
from loaders.datasets import MP3SliceDataset


@pytest.fixture
def dataset_params(cfg: DictConfig) -> MusicDatasetParameters:
    return MusicDatasetParameters.from_cfg(cfg)


@pytest.fixture
def dataset(cfg: DictConfig) -> MP3SliceDataset:
    return MP3SliceDataset.from_cfg(cfg)  # type: ignore


def test_MP3SliceDataset_init(
    dataset: MP3SliceDataset, dataset_params: MusicDatasetParameters
) -> None:
    assert dataset.dataset_params == dataset_params
    assert dataset.preload == dataset_params.preload  # type: ignore
    assert dataset.slice_length == dataset_params.slice_length  # type: ignore
    assert dataset.sample_rate == dataset_params.sample_rate  # type: ignore
    assert dataset.device == dataset_params.device  # type: ignore


def test_MP3SliceDataset_generate_data(dataset: MP3SliceDataset) -> None:
    dataset._generate_data()  # type: ignore
    assert len(dataset.buffer["slice"]) > 0


def test_MP3SliceDataset_resample_if_necessary(
    dataset: MP3SliceDataset,
) -> None:
    signal = torch.randn(2, 44100)
    resampled_signal = dataset._resample_if_necessary(signal, 22050)
    assert resampled_signal.shape[1] == 88200


def test_MP3SliceDataset_mix_down_if_necessary(
    dataset_params: MusicDatasetParameters,
) -> None:
    dataset = MP3SliceDataset(dataset_params)
    signal = torch.randn(2, 44100)
    mixed_down_signal = dataset._mix_down_if_necessary(signal)  # type: ignore
    assert mixed_down_signal.shape[0] == 1


def test_MP3SliceDataset_right_pad_if_necessary(
    dataset_params: MusicDatasetParameters,
) -> None:
    dataset = MP3SliceDataset(dataset_params)
    signal = torch.randn(1, 1023)
    padded_signal = dataset._right_pad_if_necessary(signal)  # type: ignore
    assert padded_signal.shape[1] == 1024


def test_MP3SliceDataset_getitem(dataset_params: MusicDatasetParameters) -> None:
    dataset = MP3SliceDataset(dataset_params)
    item = dataset[0]
    assert isinstance(item, dict)
    assert item["slice"].shape[0] == 1
    assert item["slice"].shape[1] == dataset_params.slice_length
    assert item["track_idx"] == 0


def test_MP3SliceDataset_len(dataset_params: MusicDatasetParameters) -> None:
    dataset = MP3SliceDataset(dataset_params)
    assert len(dataset) == len(dataset.buffer["slice"])
