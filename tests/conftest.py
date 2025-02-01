import os
import pytest

from loaders.datasets.music import MP3SliceDataset
from loaders.datasets.quantized import QuantizedUint8MusicDataset


@pytest.fixture
def mp3_dataset() -> MP3SliceDataset:
    return MP3SliceDataset(
        data_path="tests/data", sample_rate=44100, slice_length=1024, device="cpu"
    )


@pytest.fixture
def music_quant_test_dataset() -> QuantizedUint8MusicDataset:
    return QuantizedUint8MusicDataset(
        data_path=os.path.join("tests", "data"), slice_length=1024, sample_rate=44100
    )
