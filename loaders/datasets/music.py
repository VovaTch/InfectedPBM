from dataclasses import dataclass
import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import tqdm
import torchaudio
from torchaudio import load

from utils.logger import logger


@dataclass
class DataPoint:
    """
    DataPoint represents a single data point in a music dataset.

    Attributes:
        _id (int): The unique identifier of the data point.
        slice (torch.Tensor): The tensor representing the slice of the music track.
        slice_file_name (str): The file name of the slice.
        track_name (str): The name of the music track.
        track_idx (int): The index of the track in the dataset.
        slice_idx (int): The index of the slice within the track.
        slice_init_idx (int): The initial index of the slice in the track.
        slice_init_time (float): The initial time of the slice in the track.
    """

    _id: int
    slice: torch.Tensor
    slice_file_path: str | None
    track_path: str
    track_idx: int
    slice_idx: int
    slice_init_idx: int
    slice_init_time: float

    def get_metadata(self) -> dict[str, Any]:
        """
        Returns the metadata of the data point as a dictionary.

        Returns:
            dict[str, Any]: The metadata of the data point.
        """
        return {
            "_id": self._id,
            "slice_file_path": self.slice_file_path,
            "track_path": self.track_path,
            "track_idx": self.track_idx,
            "slice_idx": self.slice_idx,
            "slice_init_idx": self.slice_init_idx,
            "slice_init_time": self.slice_init_time,
        }


class MP3SliceDataset(Dataset):
    """
    Basic music slice dataset. Loads .mp3 files from a folder, converts them into one channel of long tensors,
    stores them with metadata that includes indices, time stamps, and file names. Can be extended to include
    also Mel-Spectrograms.
    """

    buffer: list[DataPoint]

    def __init__(
        self,
        data_path: str,
        sample_rate: int,
        slice_length: int,
        device: str = "cpu",
        processed_path: str | None = None,
        save_processed: bool = False,
    ) -> None:
        """
        Initializer method

        Args:
            dataset_params (MusicDatasetParameters): Dataset parameter object
        """
        super().__init__()

        self._sample_rate = sample_rate
        self._data_path = data_path
        self._slice_length = slice_length
        self._device = device

        if processed_path and os.path.exists(processed_path):
            logger.info(f"Loading processed data from {processed_path}...")
            self.buffer = self._load_data(processed_path)
            logger.info("Loaded processed data")
            return

        logger.info(f"Generating data from {data_path}...")
        self.buffer = self._generate_data(data_path, processed_path)
        logger.info("Generated data")

        if save_processed and processed_path:
            os.makedirs(processed_path, exist_ok=True)
            self._dump_data(processed_path)
            logger.info(f"Saved processed data to {processed_path}")

    def _generate_data(
        self, data_path: str, processed_path: str | None
    ) -> list[DataPoint]:
        """
        Helper method for loading data from all music files in a folder into the buffer.
        """
        # Create mp3 file list
        self.file_list = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".mp3"):
                    self.file_list.append(os.path.join(root, file))
        self.file_list = sorted(self.file_list)

        # Initialize mel spectrogram
        running_idx = 0
        data_collection = []
        for idx, file in tqdm.tqdm(
            enumerate(self.file_list), "Loading data from files..."
        ):
            file_data = self._load_data_from_track(
                file, idx, processed_path, running_idx
            )
            running_idx += len(file_data)
            data_collection += file_data
        return data_collection

    def _load_data_from_track(
        self, file: str, track_idx: int, processed_path: str | None, running_idx: int
    ) -> list[DataPoint]:
        """
        Loads and processes audio data from a given track file.

        Args:
            file (str): The path to the audio file to be loaded.
            track_idx (int): The index of the track.
            processed_path (str | None): The path where processed slices will be saved, or None if not saving.
            running_idx (int): The running index of the data points.

        Returns:
            list[DataPoint]: A list of DataPoint objects containing the processed audio slices and metadata.
        """
        print(f"loading {file}")
        long_data, sr = load(file, format="mp3")  # type: ignore
        long_data = self._resample_if_necessary(long_data, sr)
        long_data = self._mix_down_if_necessary(long_data)
        long_data = self._right_pad_if_necessary(long_data)
        slices = long_data.contiguous().view((-1, 1, self._slice_length))
        slice_file_name = self._get_slices_file_name_from_path(file, processed_path)

        data_points: list[DataPoint] = []
        for idx in range(slices.shape[0]):
            data_point = DataPoint(
                _id=running_idx + idx,
                slice=slices[idx].to(self._device),
                slice_file_path=slice_file_name,
                track_path=file,
                track_idx=track_idx,
                slice_idx=idx,
                slice_init_idx=idx * self._slice_length,
                slice_init_time=idx * self._slice_length / self._sample_rate,
            )

            data_points.append(data_point)

        return data_points

    @staticmethod
    def _get_slices_file_name_from_path(
        file_path: str, processed_path: str | None
    ) -> str | None:
        """
        Generate a new .pt file path for slices based on the given file path.

        Args:
            file_path (str): The path of the original file.

        Returns:
            str: The new file path with the modified filename.
        """
        if not processed_path:
            return None
        _, file = os.path.split(file_path)
        name, _ = os.path.splitext(file)
        new_filename = f"slices_{name}.pt"
        new_filename = new_filename.replace(" ", "_")
        return os.path.join(processed_path, new_filename)

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
        if sampling_rate != self._sample_rate:
            resampler = torchaudio.transforms.Resample(sampling_rate, self._sample_rate)
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
        if length_signal % self._slice_length != 0:
            num_missing_samples = (
                self._slice_length - length_signal % self._slice_length
            )
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        return signal

    def _dump_data(self, path: str) -> None:
        """
        Saves the data in a designated folder path

        Args:
            path (str): Saved data folder path
        """
        # Save metadata
        os.makedirs(path, exist_ok=True)
        metadata_path = os.path.join(path, "_metadata.json")
        metadata_for_json = [buffer_data.get_metadata() for buffer_data in self.buffer]
        with open(metadata_path, "w") as f:
            json.dump(metadata_for_json, f, indent=4)

        # Save slice data
        file_mapping = {
            buffer_data.track_path: buffer_data.slice_file_path
            for buffer_data in self.buffer
        }

        for track_file, slice_file in tqdm.tqdm(
            file_mapping.items(), "Saving slices as .pt files..."
        ):
            extracted_data_points = list(
                filter(lambda x: x.track_path == track_file, self.buffer)
            )
            sorted_data_points = sorted(
                extracted_data_points, key=lambda x: x.slice_idx
            )
            aggregate_slices = torch.stack(
                [data_point.slice for data_point in sorted_data_points], dim=0
            )
            if slice_file is not None:
                torch.save(aggregate_slices, slice_file)

    def _load_data(self, path: str) -> list[DataPoint]:
        """
        Loads the data from a designated folder path

        Args:
            path (str): Loaded data folder path
        """

        loaded_data = []

        json_file_path = os.path.join(path, "_metadata.json")
        with open(json_file_path, "r") as f:
            metadata: list[dict[str, Any]] = json.load(f)

        slice_files: list[str] = list(
            sorted({data_point["slice_file_path"] for data_point in metadata})
        )
        for slice_file in tqdm.tqdm(slice_files, "Loading slices..."):
            loaded_slice: torch.Tensor = torch.load(slice_file)
            slice_metadata = list(
                filter(lambda x: x["slice_file_path"] == slice_file, metadata)
            )
            slice_metadata = sorted(slice_metadata, key=lambda x: x["slice_idx"])
            for idx, slice_data_point in enumerate(slice_metadata):
                chopped_slice = loaded_slice[idx, ...]
                augmented_slice_data = {
                    **slice_data_point,
                    "slice": chopped_slice,
                }
                new_data_point = DataPoint(**augmented_slice_data)  # type: ignore
                loaded_data.append(new_data_point)

        logger.info("Parsed metadata and the slices to the buffer")
        return loaded_data

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
        data_point = self.buffer[index]
        return {**data_point.get_metadata(), "slice": data_point.slice.to(self._device)}

    def __len__(self) -> int:
        """
        Dataset length getter method

        Returns:
            int: Dataset length
        """
        return len(self.buffer)
