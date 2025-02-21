import os
import tempfile

from loaders.datasets.music import MP3SliceDataset


def test_dataset_data_length(mp3_dataset: MP3SliceDataset) -> None:
    assert len(mp3_dataset) == 734


def test_dataset_get_data(mp3_dataset: MP3SliceDataset) -> None:
    data = mp3_dataset[0]
    assert data["slice_idx"] == 0
    assert data["track_path"] == os.path.join("tests", "data", "track_1.mp3")
    assert data["slice"].shape == (1, 1024)


def test_dataset_process_files(
    mp3_dataset: MP3SliceDataset,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        mp3_dataset.buffer = mp3_dataset._generate_data(
            os.path.join("tests", "data"), temp_dir
        )
        mp3_dataset._dump_data(temp_dir)
        mp3_dataset.buffer = mp3_dataset._load_data(temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "_metadata.json"))
        assert len(mp3_dataset.buffer) == 734
        assert mp3_dataset.buffer[0].slice.shape == (1, 1024)
