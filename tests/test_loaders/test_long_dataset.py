import random
import pytest
from pytest import MonkeyPatch
from omegaconf import OmegaConf
import torch

from loaders.datasets.long import MP3LongDataset


@pytest.fixture
def dataset() -> MP3LongDataset:
    cfg = OmegaConf.create(
        {
            "dataset": {
                "dataset_type": "long",
                "data_module_type": "music",
                "data_dir": "data",
                "slice_length": 1024,
                "preload": False,
                "sample_rate": 44100,
                "preload_data_dir": "data/long_slices",
                "device": "cpu",
            }
        }
    )
    return MP3LongDataset.from_cfg(cfg)


def test_from_cfg(dataset: MP3LongDataset) -> None:

    assert dataset.steps_per_epoch == int(1e3)
    assert dataset.sample_rate == 44100
    assert dataset.device == "cpu"
    assert dataset.preload == False


def test_get_item(monkeypatch: MonkeyPatch, dataset: MP3LongDataset) -> None:
    dataset.dataset_params.slice_length = random.randint(1, 1024)

    def mock_randint(low: int, high: int, size: int) -> torch.Tensor:
        return torch.tensor([0])

    monkeypatch.setattr(torch, "randint", mock_randint)
    item = dataset.__getitem__(0)

    assert item["slice"].shape[1] == dataset.dataset_params.slice_length
    assert item["track_idx"] == 0
