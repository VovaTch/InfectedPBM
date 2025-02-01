from __future__ import annotations
from dataclasses import dataclass
import logging
import os

import tqdm
import torch
from torch.utils.data import Dataset
from torchaudio import load, transforms

from utils.waveform_tokenization import quantize_waveform_256


@dataclass
class FileDataPoint:
    """
    FileDataPoint is a data class that represents a data point in a dataset.

    Attributes:
        _id (int): The unique identifier for the data point.
        origin_path (str): The original file path of the data point.
        processed_path (str | None): The processed file path of the data point, if any.
        data (torch.Tensor): The tensor data associated with the data point.
        index_start (int): The starting index of the data point.
        index_end (int): The ending index of the data point.

    Methods:
        to(device: torch.device) -> Self:
            Transfers the tensor data to the specified device and returns a new FileDataPoint instance with the
            updated data.
    """

    _id: int
    origin_path: str
    processed_path: str | None
    data: torch.Tensor
    index_start: int
    index_end: int

    def to(self, device: torch.device | str) -> FileDataPoint:
        """
        Transfers the FileDataPoint instance to the specified device.

        Args:
            device (torch.device): The device to which the data should be transferred.

        Returns:
            Self: A new instance of FileDataPoint with data transferred to the specified device.
        """
        return FileDataPoint(
            _id=self._id,
            origin_path=self.origin_path,
            processed_path=self.processed_path,
            data=self.data.to(device),
            index_start=self.index_start,
            index_end=self.index_end,
        )


class QuantizedUint8MusicDataset(Dataset):
    """
    Dataset for converting mp3 music files into uint8 tensors, used for entropy prediction.
    """

    _data: list[FileDataPoint] = []

    def __init__(
        self,
        data_path: str,
        slice_length: int,
        sample_rate: int = 44100,
        device: str = "cpu",
        processed_path: str | None = None,
        save_processed: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initializes the dataset loader.

        Args:
            data_path (str): Path to the dataset.
            slice_length (int): Length of each audio slice.
            sample_rate (int, optional): Sampling rate of the audio. Defaults to 44100.
            device (str, optional): Device to be used for processing ('cpu' or 'cuda'). Defaults to "cpu".
            processed_path (str | None, optional): Path to preprocessed data. If provided and exists,
                preprocessed data will be loaded. Defaults to None.
            save_processed (bool, optional): Whether to save the preprocessed data. Defaults to False.
            logger (logging.Logger | None, optional): Logger for logging information. Defaults to None.

        Returns:
            None
        """

        self._data_path = data_path
        self._sample_rate = sample_rate
        self._slice_length = slice_length
        self._device = device
        self._logger = logger

        # Preprocessed data is available
        if processed_path and os.path.exists(processed_path):
            self._data = self._load_preprocessed(processed_path)
            return

        # Preprocessed data is not available
        self._data = self._preprocess(data_path, processed_path)

        # If saving preprocessed data
        if save_processed and processed_path:
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            self._save_preprocessed(processed_path)

    def _preprocess(
        self, data_path: str, preprocessed_path: str | None
    ) -> list[FileDataPoint]:
        """
        Preprocess the data and return a list of FileDataPoint objects.

        Args:
            data_path (str): Path to the data
            preprocessed_path (str | None): Path to save preprocessed data or None

        Returns:
            list[FileDataPoint]: List of FileDataPoint objects
        """
        data: list[FileDataPoint] = []
        running_inner_idx = 0
        running_outer_idx = 0
        for root, _, files in tqdm.tqdm(
            os.walk(data_path),
            desc=f"Preprocessing mp3 data from {data_path}...",
        ):
            for file in sorted(files):
                if not file.endswith(".mp3"):
                    continue
                file_path = os.path.join(root, file)
                data_point = self._process_file(
                    file_path, preprocessed_path, running_outer_idx, running_inner_idx
                )
                running_outer_idx += 1
                data.append(data_point)
                running_inner_idx = data_point.index_end

        if self._logger:
            self._logger.info(f"Preprocessed {len(data)} files")

        return data

    def _load_preprocessed(self, preprocessed_path: str) -> list[FileDataPoint]:
        """
        Load preprocessed data from a file.

        Args:
            preprocessed_path (str): Path to the preprocessed data

        Returns:
            list[FileDataPoint]: List of FileDataPoint objects
        """
        data: list[FileDataPoint] = []
        for root, _, files in tqdm.tqdm(
            os.walk(preprocessed_path),
            desc=f"Loading preprocessed data from {preprocessed_path}...",
        ):
            for file in files:
                if not file.endswith(".pt"):
                    continue
                file_path = os.path.join(root, file)
                data_point: FileDataPoint = torch.load(file_path, weights_only=False)
                data.append(data_point.to(self._device))

        if self._logger:
            self._logger.info(f"Loaded {len(data)} files")

        return data

    def _save_preprocessed(self, preprocessed_path: str) -> None:
        """
        Save preprocessed data to a file.

        Args:
            preprocessed_path (str): Path to save the preprocessed data
        """
        for data_point in tqdm.tqdm(
            self._data, desc=f"Saving preprocessed data to {preprocessed_path}..."
        ):
            if not data_point.processed_path:
                path_to_save = self._processed_file_path(
                    data_point.origin_path, preprocessed_path
                )
            else:
                path_to_save = data_point.processed_path
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            torch.save(
                data_point,
                path_to_save,
            )

        if self._logger:
            self._logger.info(f"Saved {len(self._data)} files")

    @staticmethod
    def _processed_file_path(file_path: str, processed_dir_path: str) -> str:
        """
        Generate the path for the processed file.

        Args:
            file_path (str): The original file path.
            processed_dir_path (str): The directory path where the processed files are stored.

        Returns:
            str: The path for the processed file with a ".pt" extension.
        """
        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0] + ".pt"
        updated_file_path = os.path.join(processed_dir_path, file_name)
        return updated_file_path

    def _process_file(
        self,
        file_path: str,
        processed_dir_path: str | None,
        file_idx: int,
        running_idx: int,
    ) -> FileDataPoint:
        """
        Processes an audio file by loading, resampling, mixing down, and converting it to uint8 format.

        Args:
            file_path (str): The path to the audio file to be processed.
            processed_dir_path (str | None): The directory path where the processed file will be saved.
            file_idx (int): The index of the file being processed.
            running_idx (int): The running index used to determine the start and end indices of the processed data.

        Returns:
            FileDataPoint: An object containing metadata and processed audio data.
        """
        long_data, sampling_rate = load(file_path)
        long_data = self._resample_if_necessary(long_data, sampling_rate)
        long_data = self._mix_down_if_necessary(long_data).squeeze(0)
        long_data_uint8 = quantize_waveform_256(long_data)
        file_len = long_data_uint8.shape[0]

        return FileDataPoint(
            _id=file_idx,
            origin_path=file_path,
            processed_path=(
                self._processed_file_path(file_path, processed_dir_path)
                if processed_dir_path
                else None
            ),
            data=long_data_uint8.to(self._device),
            index_start=running_idx,
            index_end=running_idx + file_len - self._slice_length,
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
        if sampling_rate != self._sample_rate:
            resampler = transforms.Resample(sampling_rate, self._sample_rate)
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

    def __len__(self) -> int:
        return sum(
            [data_point.index_end - data_point.index_start for data_point in self._data]
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        data_points = list(
            filter(lambda x: x.index_start <= idx < x.index_end, self._data)
        )
        if not data_points:
            raise IndexError(f"Index {idx} out of range")
        data_point = data_points[0]
        raw_data = data_point.data[
            idx
            - data_point.index_start : idx
            + self._slice_length
            - data_point.index_start
        ]
        return {
            "slice": raw_data.to(self._device),
            "file_id": torch.tensor(data_point._id, dtype=torch.int).to(self._device),
            "file_name": data_point.origin_path,
        }
