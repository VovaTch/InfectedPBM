import os
import tempfile

from loaders.datasets.latent import LatentSliceDataset
from models.models.multi_level_vqvae.ml_vqvae import MultiLvlVQVariationalAutoEncoder


def test_dataset_data_length(lvl1_latent_dataset: LatentSliceDataset) -> None:
    assert len(lvl1_latent_dataset) == 368


def test_dataset_get_data(lvl1_latent_dataset: LatentSliceDataset) -> None:
    data = lvl1_latent_dataset[0]
    assert data["latent_idx"] == 0
    assert data["slice_path"] == os.path.join(
        "tests", "data_slices", "slices_track_1.pt"
    )
    assert data["latent"].shape == (16, 4)


def test_dataset_process_files(
    lvl1_latent_dataset: LatentSliceDataset,
    vqvae_basic: MultiLvlVQVariationalAutoEncoder,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        lvl1_latent_dataset.buffer = lvl1_latent_dataset._generate_data(
            os.path.join("tests", "data_slices"), temp_dir, vqvae_basic
        )
        lvl1_latent_dataset._dump_data(temp_dir)
        lvl1_latent_dataset.buffer = lvl1_latent_dataset._load_data(temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "_metadata.json"))
        assert len(lvl1_latent_dataset.buffer) == 368
        assert lvl1_latent_dataset.buffer[0].latent.shape == (16, 4)
