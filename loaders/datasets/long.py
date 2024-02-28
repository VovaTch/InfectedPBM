import json
import os
from typing import Any
from typing_extensions import Self
from matplotlib.pyplot import step
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F
import tqdm

from common import registry
from tests.test_loaders.test_datasets import dataset_params
from utils.containers import MusicDatasetParameters


@registry.register_dataset("long")
class MP3LongDataset(Dataset):
    """
    Dataset for chaining entire MP3 files in a single tensor. Used for unsupervised learning
    from pure waveform, possibly training a better tokenizer than the structured slice one.
    """

    dataset_params: MusicDatasetParameters
    buffer: dict[str, Any] = {}
    file_list: list[str] = []

    def __init__(
        self, dataset_params: MusicDatasetParameters, steps_per_epoch: int = int(1e3)
    ) -> None:
        super().__init__()
        self.dataset_params = dataset_params
        self.preload = dataset_params.preload
        self.sample_rate = dataset_params.sample_rate
        self.device = dataset_params.device
        self.steps_per_epoch = steps_per_epoch

        # Check if the data is loaded
        track_path = os.path.join(dataset_params.data_dir, "tracks")
        if os.path.exists(track_path):
            self._load_data(dataset_params.data_dir)
        else:
            if not os.path.exists(dataset_params.data_dir):
                os.makedirs(dataset_params.data_dir)
            self._generate_data()
            self._dump_data(dataset_params.data_dir)

    def _load_data(self, path: str) -> None:
        """
        Load the data from the given path.

        Args:
            path (str): The path to the data.
        """

        json_file_path = os.path.join(path, "metadata.json")
        with open(json_file_path, "r") as f:
            metadata = json.load(f)

        # Reorganize the metadata file as a buffer
        for datapoint in metadata:
            for key, value in datapoint.items():
                # Convert to the correct device
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)

                # Input in the buffer
                if key not in self.buffer:
                    self.buffer[key] = [value]
                else:
                    self.buffer[key] += [value]

        # Load slice into the buffer
        for file_path in tqdm.tqdm(
            self.buffer["slice_file_name"], "Loading slices to buffer..."
        ):
            self.buffer["slice"] += [torch.load(os.path.join(path, file_path))]

    def _generate_data(self) -> None:
        """
        Generate the data.

        This method generates the data by creating a file list from the specified data directory,
        loading the data into a buffer, and appending the loaded data to the buffer.
        """
        # Create file list
        self.file_list = []
        for subdir, _, files in os.walk(self.dataset_params.data_dir):
            for file in files:
                if file.endswith(".mp3"):
                    self.file_list.append(os.path.join(subdir, file))

        # Load data into buffer
        for idx, file in tqdm.tqdm(
            enumerate(self.file_list), "Loading data from files..."
        ):
            file_data = self._load_data_from_track(file, idx)

            # Append to the buffer
            for key, value in file_data.items():
                if key in self.buffer:
                    self.buffer[key] += value
                else:
                    self.buffer[key] = value

    def _load_data_from_track(self, file: str, idx: int) -> dict[str, Any]:
        """
        Load data from a track file.

        Args:
            file (str): The path to the track file.
            idx (int): The index of the track.

        Returns:
            dict[str, Any]: A dictionary containing the loaded data, including the slice, slice file name,
                            track name, and track index.
        """
        print(f"Loading {file}...")
        long_data, sr = torchaudio.load(file, format="mp3")  # type: ignore
        long_data = self._resample_if_necessary(long_data, sr)
        long_data = self._mix_down_if_necessary(long_data)

        # Handle track name
        track_file_name = "long-slices_" + file.split("/")[-1][:-4] + ".pt"
        track_file_name = track_file_name.replace(" ", "_")

        return {
            "slice": [long_data],
            "slice_file_name": [track_file_name],
            "track_name": [file.split("/")[-1]],
            "track_idx": [idx],
        }

    def _dump_data(self, path: str) -> None:
        """
        Dump the data to the given path.

        Args:
            path (str): The path to the data.
        """

        # Create track folder
        track_path = os.path.join(path, "long_slices")
        if not os.path.exists(track_path):
            os.makedirs(track_path)

        # Make metadata folder
        metadata_path = os.path.join(path, "metadata_long.json")
        metadata_for_json = [
            {key: value[idx] for (key, value) in self.buffer.items() if key != "slice"}
            for idx in range(len(self.buffer["slice_idx"]))
        ]
        with open(metadata_path, "w") as f:
            json.dump(metadata_for_json, f, indent=4)

        # Save long-slice files
        for idx, slice in enumerate(self.buffer["slice"]):
            torch.save(
                slice,
                os.path.join(track_path, self.buffer["slice_file_name"][idx]),
            )

    def _resample_if_necessary(
        self, signal: torch.Tensor, sampling_rate: int
    ) -> torch.Tensor:
        """
        Helper method to change the sampling rate of a music track

        Args:
            signal (torch.Tensor): Music slice, expects `C x L` size
            sampling_rate (int): Sampling rate, default for MP3 is 44100.

        Returns:
            torch.Tensor: Resampled signal
        """
        if sampling_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sampling_rate, self.sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Helper method to merge down music channels, currently the code doesn't support more than 1 channel.

        Args:
            signal (torch.Tensor): Signal to be mixed down, shape `C x L`

        Returns:
            torch.Tensor: Mixed-down signal, shape `1 x L`
        """

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _right_pad_if_necessary(
        self, signal: torch.Tensor, slice_length: int
    ) -> torch.Tensor:
        """
        Helper function aimed to keep all the slices at a constant size, pad with 0 if the slice is too short.

        Args:
            signal (torch.Tensor): Input slice, shape `1 x L*`
            slice_length (int): Desired length of the slice after padding.

        Returns:
            torch.Tensor: Output slice, shape `1 x L` padded with zeroes.
        """
        length_signal = signal.shape[1]
        if length_signal % slice_length != 0:
            num_missing_samples = slice_length - length_signal % slice_length
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        return signal

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.steps_per_epoch

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        """
        Retrieves an item from the dataset.

        Args:
            _: The index of the item to retrieve. This argument is ignored.

        Returns:
            A dictionary containing the following keys:
            - "slice": The slice tensor.
            - "slice_file_name": The file name of the slice.
            - "track_name": The name of the track.
            - "track_idx": The index of the track.
        """
        random_track_idx = torch.randint(0, len(self.buffer["slice"]), (1,)).item()
        random_idx = torch.randint(
            0, self.buffer["slice"][random_track_idx].shape[1], (1,)
        ).item()
        if (
            random_idx + self.dataset_params.slice_length
            > self.buffer["slice"][random_track_idx].shape[1]
        ):
            slice = self.buffer["slice"][:, random_track_idx:]
            slice = self._right_pad_if_necessary(
                slice, self.dataset_params.slice_length
            )

        else:
            slice = self.buffer["slice"][
                random_track_idx,
                random_idx : random_idx + self.dataset_params.slice_length,
            ]

        return {
            "slice": slice,
            "slice_file_name": self.buffer["slice_file_name"][random_track_idx],
            "track_name": self.buffer["track_name"][random_track_idx],
            "track_idx": self.buffer["track_idx"][random_track_idx],
        }

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create an instance of the class from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Returns:
            Self: An instance of the class.

        """
        dataset_params = MusicDatasetParameters.from_cfg(cfg)
        dataset_cfg: dict[str, Any] = cfg.dataset
        steps_per_epoch = dataset_cfg.get("steps_per_epoch", int(1e3))
        return cls(dataset_params, steps_per_epoch)
