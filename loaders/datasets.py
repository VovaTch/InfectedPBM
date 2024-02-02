import json
import os
from typing import Any, Protocol
from typing_extensions import Self
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import tqdm

from utils.containers import MusicDatasetParameters
from common import registry


class MusicDataset(Protocol):
    """
    Basic music dataset protocol
    """

    dataset_params: MusicDatasetParameters
    buffer: dict[str, Any] = {}  # Data buffer dictionary

    def __init__(self, dataset_params: MusicDatasetParameters) -> None:
        """
        Initializer method

        Args:
            dataset_params (MusicDatasetParameters): Dataset parameter object
        """
        ...

    def _dump_data(self, path: str) -> None:
        """
        Saves the data in a designated folder path

        Args:
            path (str): Saved data folder path
        """
        ...

    def _load_data(self, path: str) -> None:
        """
        Loads the data from a designated folder path

        Args:
            path (str): Loaded data folder path
        """
        ...

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Standard dataset object __getitem__ method

        Args:
            index (int): Index of the data-point

        Returns:
            dict[str, Any]: Dictionary item from the dataset, collected values with a
            collate_fn function from Pytorch
        """
        ...

    def __len__(self) -> int:
        """
        Dataset length getter method

        Returns:
            int: Dataset length
        """
        ...

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Utility method to parse dataset parameters from a configuration dictionary

        Args:
            cfg (DictConfig): configuration dictionary

        Returns:
            MusicDataset: dataset object
        """
        ...


@registry.register_dataset("mp3_slice")  # type: ignore
class MP3SliceDataset(Dataset):
    """
    Basic music slice dataset. Loads .mp3 files from a folder, converts them into one channel of long tensors,
    stores them with metadata that includes indices, time stamps, and file names. Can be extended to include
    also Mel-Spectrograms.
    """

    dataset_params: MusicDatasetParameters
    buffer: dict[str, Any] = {}

    def __init__(self, dataset_params: MusicDatasetParameters) -> None:
        """
        Initializer method

        Args:
            dataset_params (MusicDatasetParameters): Dataset parameter object
        """
        super().__init__()

        self.dataset_params = dataset_params
        self.preload = dataset_params.preload
        self.slice_length = dataset_params.slice_length
        self.sample_rate = dataset_params.sample_rate
        self.device = dataset_params.device

        # Check if the data is loaded
        slices_path = os.path.join(dataset_params.data_dir, "slices")
        if os.path.exists(slices_path):
            self._load_data(dataset_params.data_dir)
        else:
            if not os.path.exists(dataset_params.data_dir):
                os.makedirs(dataset_params.data_dir)
            self._generate_data()
            self._dump_data(dataset_params.data_dir)

    def _generate_data(self) -> None:
        """
        Helper method for loading data from all music files in a folder into the buffer.
        """
        # Create mp3 file list
        self.file_list = []
        for subdir, _, files in os.walk(self.dataset_params.data_dir):
            for file in files:
                if file.endswith(".mp3"):
                    self.file_list.append(os.path.join(subdir, file))

        # Initialize mel spectrogram
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

    def _load_data_from_track(self, file: str, track_idx: int) -> dict[str, Any]:
        """
        Helper method for loading data and metadata from file into a dictionary.

        Args:
            file (str): File path, supported format is .mp3 currently.
            track_idx (int): Track index

        Returns:
            dict[str, Any]: Output dictionary contains the slice data and metadata from a file.
        """
        print(f"loading {file}")
        long_data, sr = torchaudio.load(file, format="mp3")  # type: ignore
        long_data = self._resample_if_necessary(long_data, sr)
        long_data = self._mix_down_if_necessary(long_data)
        long_data = self._right_pad_if_necessary(long_data)
        slices = long_data.view((-1, 1, self.slice_length)).half()
        slice_file_name = "slices_" + file.split("/")[-1][:-4] + ".pt"
        slice_file_name = slice_file_name.replace(" ", "_")

        # Make the data fit into the buffer
        slices_list = [ind_slice.to(self.device) for ind_slice in slices]
        track_name_list = [file.split("/")[-1] for _ in range(slices.shape[0])]
        track_idx_list = [track_idx for _ in range(slices.shape[0])]
        slice_file_name_list = [slice_file_name for _ in range(slices.shape[0])]
        slice_idx = [idx for idx in range(slices.shape[0])]
        slice_init_idx = [idx * self.slice_length for idx in range(slices.shape[0])]
        slice_init_time = [
            idx * self.slice_length / self.sample_rate for idx in range(slices.shape[0])
        ]

        # Aggregate all data in a dictionary
        data = {
            "slice": slices_list,
            "slice_file_name": slice_file_name_list,
            "track_name": track_name_list,
            "track_idx": track_idx_list,
            "slice_idx": slice_idx,
            "slice_init_idx": slice_init_idx,
            "slice_init_time": slice_init_time,
        }

        return data

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

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Helper function aimed to keep all the slices at a constant size, pad with 0 if the slice is too short.

        Args:
            signal (torch.Tensor): Input slice, shape `1 x L*`

        Returns:
            torch.Tensor: Output slice, shape `1 x L` padded with zeroes.
        """
        length_signal = signal.shape[1]
        if length_signal % self.slice_length != 0:
            num_missing_samples = self.slice_length - length_signal % self.slice_length
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        return signal

    def _dump_data(self, path: str) -> None:
        """
        Saves the data in a designated folder path

        Args:
            path (str): Saved data folder path
        """

        # Make slice folder
        slice_data_path = os.path.join(path, "slices")
        if not os.path.exists(slice_data_path):
            os.makedirs(slice_data_path)

        # Make metadata folder
        metadata_path = os.path.join(path, "metadata.json")
        metadata_for_json = [
            {key: value[idx] for (key, value) in self.buffer.items() if key != "slice"}
            for idx in range(len(self.buffer["slice_idx"]))
        ]
        with open(metadata_path, "w") as f:
            json.dump(metadata_for_json, f, indent=4)

        for file in tqdm.tqdm(self.file_list, "Saving slices as .pt files..."):
            aggregate_slices = [
                self.buffer["slice"][idx]
                for idx in range(len(self.buffer["slice"]))
                if self.buffer["track_name"][idx] == file.split("/")[-1]
            ]
            aggregate_slices_torch = torch.stack(aggregate_slices, dim=0)
            slice_file_name = "slices_" + file.split("/")[-1][:-4] + ".pt"
            slice_file_name = slice_file_name.replace(" ", "_")
            slice_file_path = os.path.join(slice_data_path, slice_file_name)

            # Saving this as a .pt file
            torch.save(aggregate_slices_torch, slice_file_path)

    def _load_data(self, path: str) -> None:
        """
        Loads the data from a designated folder path

        Args:
            path (str): Loaded data folder path
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

        # Load the slices
        slice_file_paths = self._create_slice_file_list(path, metadata)
        for file_path in tqdm.tqdm(slice_file_paths, "Parsing slices to buffer..."):
            slices = torch.load(file_path)
            self._parse_slices_to_buffer(slices.float())

        print("Parsed metadata and the slices to the buffer")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Standard dataset object __getitem__ method

        Args:
            index (int): Index of the data-point

        Returns:
            *   dict[str, Any]: Dictionary item from the dataset, collected values with a
                collate_fn function from Pytorch. The expected slice output from a dataloader is:
                -   `slice`: tensor size `BS x 1 x L`
        """

        datapoint = {key: value[index] for (key, value) in self.buffer.items()}
        for value in datapoint.values():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)

        return datapoint

    @staticmethod
    def _create_slice_file_list(
        root_data_path: str, metadata: list[dict[str, Any]]
    ) -> list[str]:
        """
        Helper method to attach the file name to each slice

        Args:
            root_data_path (str): .mp3 file folder
            metadata (list[dict[str, Any]]): metadata list

        Returns:
            list[str]: list of music file paths, at the length of the metadata list
        """
        slice_file_names = []
        current_file_name = None
        for data_point in metadata:
            if data_point["slice_file_name"] != current_file_name:
                slice_file_names.append(data_point["slice_file_name"])
                current_file_name = data_point["slice_file_name"]

        # Create the paths
        slice_file_paths = [
            os.path.join(root_data_path, "slices", ind_file_name)
            for ind_file_name in slice_file_names
        ]
        return slice_file_paths

    def _parse_slices_to_buffer(self, slices: torch.Tensor) -> None:
        """
        Helper method to fit the slices into the buffer

        Args:
            slices (torch.Tensor): Music slices
        """
        for slice in slices:
            if "slice" not in self.buffer:
                self.buffer["slice"] = [slice]
            else:
                self.buffer["slice"] += [slice]

    def __len__(self) -> int:
        """
        Dataset length getter method

        Returns:
            int: Dataset length
        """
        return len(self.buffer["slice"])

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Utility method to parse dataset parameters from a configuration dictionary

        Args:
            cfg (dict[str, Any]): configuration dictionary

        Returns:
            MusicDataset: dataset object
        """
        dataset_params = MusicDatasetParameters.from_cfg(cfg)
        return cls(dataset_params)


@registry.register_dataset("test_mp3_slice")  # type: ignore
class TestDataset(MP3SliceDataset):  # type: ignore
    """
    Dataset object for testing, which inherits from the slice dataset and modifies the __len__
    method to return a small number rather than the entire dataset.
    """

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        Returns:
            int: 6, it is short and to the point.
        """
        return 6

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
        return cls(dataset_params)
