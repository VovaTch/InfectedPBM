import os
import tempfile
import torch

from loaders.datasets.quantized import QuantizedUint8MusicDataset


def test_dataset_data_length(
    music_quant_test_dataset: QuantizedUint8MusicDataset,
) -> None:
    assert len(music_quant_test_dataset._data) == 3


def test_dataset_length(music_quant_test_dataset: QuantizedUint8MusicDataset) -> None:
    assert len(music_quant_test_dataset) == 200000 + 250000 + 300000 - 3 * 1024


def test_dataset_getitem(music_quant_test_dataset: QuantizedUint8MusicDataset) -> None:
    data_point = music_quant_test_dataset[0]
    assert set(data_point.keys()) == {"slice", "file_id", "file_name"}
    assert isinstance(data_point["slice"], torch.Tensor)
    assert list(data_point["slice"].shape) == [1024]
    assert data_point["file_id"] == 0
    assert data_point["file_name"] == os.path.join("tests", "data", "track_1.mp3")


def test_dataset_process_files(
    music_quant_test_dataset: QuantizedUint8MusicDataset,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        music_quant_test_dataset._data = music_quant_test_dataset._preprocess(
            os.path.join("tests", "data"), temp_dir
        )
        music_quant_test_dataset._save_preprocessed(temp_dir)
        music_quant_test_dataset._data = music_quant_test_dataset._load_preprocessed(
            temp_dir
        )
        assert len(music_quant_test_dataset._data) == 3
        assert music_quant_test_dataset[0]["file_id"] == 0
        assert music_quant_test_dataset[0]["file_name"] == os.path.join(
            "tests", "data", "track_1.mp3"
        )
        assert isinstance(music_quant_test_dataset[0]["slice"], torch.Tensor)
        assert list(music_quant_test_dataset[0]["slice"].shape) == [1024]
