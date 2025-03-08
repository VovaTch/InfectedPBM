from dataclasses import dataclass
import json
import os
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import tqdm

from utils.logger import logger


class Tokenizer(Protocol):
    def tokenize(self, x: torch.Tensor) -> torch.Tensor: ...
    def to(self, device: str) -> "Tokenizer": ...


@dataclass
class LatentDataPoint:
    """
    LatentDataPoint is a data class that represents a single data point in a latent dataset.

    Attributes:
        _id (int): Unique identifier for the data point.
        slice_level (int | None): The level of the slice, e.g., 1 uses lvl1_vqvae.
        latent (torch.Tensor): The latent tensor associated with the data point.
        latent_path (str | None): The file path to the latent tensor.
        slice_path (str): The file path to the slice.
        slice_idx (int): The index of the slice.
        latent_idx (int): The index of the latent tensor.
        latent_init_idx (int): The initial index of the latent tensor.
        track_init_time (float): The time stamp at the original file. TODO: add

    Methods:
        get_metadata() -> dict[str, int | str | None | float]:
            Returns a dictionary containing metadata of the data point.
    """

    _id: int
    slice_level: int | None  # 1 uses lvl1_vqvae for example
    latent: torch.Tensor
    latent_path: str | None
    slice_path: str
    slice_idx: int
    latent_idx: int
    latent_init_idx: int
    track_init_time: float

    def get_metadata(self) -> dict[str, int | str | None | float]:
        """
        Retrieve metadata information as a dictionary.

        Returns:
            dict[str, int | str | None | float]: A dictionary containing metadata with the following keys:
                - "_id": The unique identifier of the object.
                - "slice_level": The level of the slice.
                - "latent_idx": The index of the latent variable.
                - "latent_init_idx": The initial index of the latent variable.
                - "track_init_time": The initial time of tracking.
                - "latent_path": The path to the latent variable.
                - "slice_path": The path to the slice.
                - "slice_idx": The index of the slice.
        """
        return {
            "_id": self._id,
            "slice_level": self.slice_level,
            "latent_idx": self.latent_idx,
            "latent_init_idx": self.latent_init_idx,
            "track_init_time": self.track_init_time,
            "latent_path": self.latent_path,
            "slice_path": self.slice_path,
            "slice_idx": self.slice_idx,
        }


class LatentSliceDataset(Dataset):
    """
    A dataset that contains the latents drawn from one level above or directly from MP3 files.
    """

    buffer: list[LatentDataPoint]

    def __init__(
        self,
        data_path: str,
        slices_per_sample: int,
        tokenizer: Tokenizer | None = None,
        tokenizing_batch_size: int = 32,
        device: str = "cpu",
        tokenizing_device: str = "cuda",
        channel_first_data: bool = False,
        sample_rate: int = 44100,
        processed_path: str | None = None,
        save_processed: bool = False,
        slice_level: int | None = None,
    ) -> None:
        """
        Initializes the dataset loader.

        Args:
            data_path (str): Path to the raw data.
            slices_per_sample (int): Number of slices per sample.
            tokenizer (Tokenizer | None, optional): Tokenizer instance for data processing. Defaults to None.
            tokenizing_batch_size (int, optional): Batch size for tokenizing. Defaults to 32.
            device (str, optional): Device to load data on. Defaults to "cpu".
            tokenizing_device (str, optional): Device to use for tokenizing. Defaults to "cuda".
            channel_first_data (bool, optional): Whether the data is channel-first. Defaults to False.
            sample_rate (int, optional): Sample rate of the data. Defaults to 44100.
            processed_path (str | None, optional): Path to the processed data. Defaults to None.
            save_processed (bool, optional): Whether to save the processed data. Defaults to False.
            slice_level (int | None, optional): Level of slicing. Defaults to None.

        Raises:
            ValueError: If no tokenizer is provided and processed data is not available.
        """
        super().__init__()

        self._data_path = data_path
        self._slices_per_sample = slices_per_sample
        self._device = device
        self._sample_rate = sample_rate
        self._tokenizing_batch_size = tokenizing_batch_size
        self._tokenizing_device = tokenizing_device
        self._slice_level = slice_level
        self._channel_first = channel_first_data

        if processed_path and os.path.exists(processed_path):
            logger.info(f"Loading processed data from {processed_path}...")
            self.buffer = self._load_data(processed_path)
            logger.info("Loaded processed data")
            return

        if tokenizer is None:
            raise ValueError(
                "A tokenizer must be provided if processed data is not available"
            )

        logger.info(f"Generating data from {data_path}...")
        self.buffer = self._generate_data(
            data_path, processed_path, tokenizer.to(self._tokenizing_device)
        )
        logger.info("Generated data")

        if save_processed and processed_path:
            os.makedirs(processed_path, exist_ok=True)
            self._dump_data(processed_path)
            logger.info(f"Saved processed data to {processed_path}")

    def _generate_data(
        self, data_path: str, processed_path: str | None, tokenizer: Tokenizer
    ) -> list[LatentDataPoint]:
        """
        Generates a list of LatentDataPoint objects by loading and processing data from files.

        Args:
            data_path (str): The path to the directory containing the data files.
            processed_path (str | None): The path to the directory where processed files are stored,
                or None if not applicable.
            tokenizer (Tokenizer): The tokenizer to be used for processing the data.

        Returns:
            list[LatentDataPoint]: A list of LatentDataPoint objects containing the processed data.
        """

        # Create slice file list
        self.file_list = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".pt"):
                    self.file_list.append(os.path.join(root, file))
        self.file_list = sorted(self.file_list)

        # Load data from files
        running_idx = 0
        data_collection = []
        for idx, file in tqdm.tqdm(
            enumerate(self.file_list), "Loading data from files..."
        ):
            file_data = self._load_data_from_file(
                file, idx, processed_path, running_idx, tokenizer
            )
            running_idx += len(file_data)
            data_collection += file_data
        return data_collection

    def _load_data_from_file(
        self,
        file: str,
        idx: int,
        processed_path: str | None,
        running_idx: int,
        tokenizer: Tokenizer,
    ) -> list[LatentDataPoint]:
        """
        Loads data from a given file, processes it, and returns a list of LatentDataPoint objects.

        Args:
            file (str): Path to the file containing the data to be loaded.
            idx (int): Index of the current file being processed.
            processed_path (str | None): Path to the processed data directory, if any.
            running_idx (int): Running index to uniquely identify data points.
            tokenizer (Tokenizer): Tokenizer object used to tokenize the data.

        Returns:
            list[LatentDataPoint]: A list of LatentDataPoint objects containing the processed data.
        """
        print(f"Loading data from {file}")
        slices: torch.Tensor = torch.load(file).to(self._tokenizing_device)
        if self._channel_first:
            slices = slices.transpose(1, 2).contiguous()
        slice_loader = DataLoader(
            TensorDataset(slices), batch_size=self._tokenizing_batch_size, shuffle=False
        )
        latent_file_name = self._get_latent_file_name_from_path(
            file, processed_path, self._slice_level
        )

        for idx, batch in enumerate(slice_loader):
            if len(batch[0]) < self._tokenizing_batch_size:
                batch_to_tokenize = F.pad(
                    batch[0], (self._tokenizing_batch_size - len(batch[0]), 0)
                )
            else:
                batch_to_tokenize = batch[0]
            if self._channel_first:
                batch_to_tokenize = batch_to_tokenize.transpose(1, 2).contiguous()

            tokenized_batch = tokenizer.tokenize(
                batch_to_tokenize
            )  # Should be size BS x SeqL x Ncb

            if idx == 0:
                collected_tokenized_slices = tokenized_batch.flatten(end_dim=1)
            else:
                collected_tokenized_slices = torch.cat(
                    (collected_tokenized_slices, tokenized_batch.flatten(end_dim=1)),
                    dim=0,
                )

        num_token_slices = collected_tokenized_slices.shape[0]
        data_points: list[LatentDataPoint] = []
        for idx in range(num_token_slices // self._slices_per_sample + 1):

            if idx < num_token_slices // self._slices_per_sample:
                latent_slice = collected_tokenized_slices[
                    idx * self._slices_per_sample : (idx + 1) * self._slices_per_sample
                ]
            elif num_token_slices % self._slices_per_sample != 0:
                partial_slice = collected_tokenized_slices[
                    idx * self._slices_per_sample : num_token_slices
                ]
                latent_slice = F.pad(
                    partial_slice,
                    (0, 0, 0, self._slices_per_sample - partial_slice.shape[0]),
                )
            else:
                continue

            data_point = LatentDataPoint(
                _id=running_idx + idx,
                slice_level=self._slice_level,
                latent=latent_slice.to(self._device),
                latent_path=latent_file_name,
                slice_path=file,
                slice_idx=idx * self._slices_per_sample,
                latent_idx=idx,
                latent_init_idx=running_idx,
                track_init_time=0,  # TODO: currently 0, might need to change later
            )
            data_points.append(data_point)

        return data_points

    @staticmethod
    def _get_latent_file_name_from_path(
        file_path: str, processed_path: str | None, latent_level: int | None = None
    ) -> str | None:
        """
        Generate a latent file name based on the given file path and processed path.

        This method constructs a new file name for a latent file by extracting the base name
        from the given file path, optionally adding a latent level prefix, and appending
        a standard suffix. The new file name is then joined with the processed path.

        Args:
            file_path (str): The original file path.
            processed_path (str | None): The directory where the new file should be saved.
            latent_level (int | None, optional): The latent level to include in the file name. Defaults to None.

        Returns:
            str | None: The constructed latent file name with the processed path,
                or None if processed_path is not provided.
        """
        if not processed_path:
            return None
        _, file = os.path.split(file_path)
        name, _ = os.path.splitext(file)
        latent_level_prefix = f"lvl{latent_level}_" if latent_level else ""
        new_filename = f"{latent_level_prefix}latents_{name}.pt"
        new_filename = new_filename.replace(" ", "_")
        return os.path.join(processed_path, new_filename)

    def _dump_data(self, path: str) -> None:
        """
        Dumps the buffered data to the specified directory.

        This method creates the specified directory if it does not exist, saves the metadata
        of the buffered data to a JSON file, and saves the latent data to .pt files.

        Args:
            path (str): The directory path where the data will be dumped.
        """
        os.makedirs(path, exist_ok=True)
        metadata_path = os.path.join(path, "_metadata.json")
        metadata_for_json = [buffer_data.get_metadata() for buffer_data in self.buffer]
        with open(metadata_path, "w") as f:
            json.dump(metadata_for_json, f)

        file_mapping = {
            buffer_data.slice_path: buffer_data.latent_path
            for buffer_data in self.buffer
        }

        for slice_file, latent_file in tqdm.tqdm(
            file_mapping.items(), "saving latents as .pt files..."
        ):
            extracted_data_points = list(
                filter(lambda x: x.slice_path == slice_file, self.buffer)
            )
            sorted_data_points = sorted(
                extracted_data_points, key=lambda x: x.latent_idx
            )
            aggregate_latents = torch.stack(
                [data_point.latent for data_point in sorted_data_points], dim=0
            )
            if latent_file is not None:
                torch.save(aggregate_latents, latent_file)

    def _load_data(self, path: str) -> list[LatentDataPoint]:
        """
        Load latent data points from the specified directory.

        Args:
            path (str): The directory path where the latent data and metadata are stored.

        Returns:
            list[LatentDataPoint]: A list of loaded latent data points.

        The function performs the following steps:
        1. Reads the metadata from a JSON file named "_metadata.json" located in the specified directory.
        2. Extracts and sorts the unique latent file paths from the metadata.
        3. Loads each latent file and associates it with its corresponding metadata.
        4. Constructs LatentDataPoint objects by combining the latent data with its metadata.
        5. Returns a list of LatentDataPoint objects.

        Note:
            - The metadata JSON file should contain a list of dictionaries,
                each representing a data point with keys such as "latent_path" and "latent_idx".
            - The latent files are expected to be in a format that can be loaded using `torch.load`.
        """
        loaded_data = []

        json_file_path = os.path.join(path, "_metadata.json")
        with open(json_file_path, "r") as f:
            metadata: list[dict[str, Any]] = json.load(f)

        latent_files: list[str] = list(
            sorted({data_point["latent_path"] for data_point in metadata})
        )
        for latent_file in tqdm.tqdm(latent_files, "Loading latents..."):
            loaded_latent: torch.Tensor = torch.load(latent_file)
            latent_metadata = list(
                filter(lambda x: x["latent_path"] == latent_file, metadata)
            )
            latent_metadata = sorted(latent_metadata, key=lambda x: x["latent_idx"])
            for idx, latent_data_point in enumerate(latent_metadata):
                chopped_latent = loaded_latent[idx, ...]
                augmented_latent_data = {**latent_data_point, "latent": chopped_latent}
                new_data_point = LatentDataPoint(**augmented_latent_data)
                loaded_data.append(new_data_point)

        logger.info("Parsed metadata and loaded latents to buffer")
        return loaded_data

    def __len__(self) -> int:
        """
        Returns the number of elements in the buffer.

        Returns:
            int: The number of elements in the buffer.
        """
        return len(self.buffer)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Retrieve a data point from the buffer at the specified index.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            dict[str, Any]: A dictionary containing the metadata of the data point
                            and its latent representation.
        """
        data_point = self.buffer[index]
        return {**data_point.get_metadata(), "latent": data_point.latent}
