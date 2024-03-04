from __future__ import annotations
import os

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import tqdm

from common import registry
from loaders.datasets.music import MP3SliceDataset
from models.music_module import MusicLightningModule
from utils.containers import MusicDatasetParameters


@registry.register_dataset("mp3_indices")
class MP3TokenizedIndicesDataset(Dataset):
    """
    MP3 dataset of tokenized indices, used to train the LLM-style models
    """

    dataset_params: MusicDatasetParameters
    tokenized_data: torch.Tensor

    def __init__(
        self,
        dataset_params: MusicDatasetParameters,
        codebook_size: int,
        index_series_length: int = 1024,
        slice_dataset: MP3SliceDataset | None = None,
        tokenizer: MusicLightningModule | None = None,
        buffer_process_batch_size: int = 32,
        epoch_size: int = 100000,
    ) -> None:
        """
        Initializes an instance of the Indices class.

        Args:
            dataset_params (MusicDatasetParameters): The parameters for the music dataset.
            codebook_size (int): The size of the codebook.
            index_series_length (int, optional): The length of the index series. Defaults to 1024.
            slice_dataset (MP3SliceDataset | None, optional): The slice dataset. Defaults to None.
            tokenizer (Tokenizer | None, optional): The tokenizer. Defaults to None.
            buffer_process_batch_size (int, optional): The batch size for buffer processing. Defaults to 32.
            epoch_size (int, optional): The size of each epoch. Defaults to 100000.
        """
        super().__init__()

        self.device = dataset_params.device
        self.tokenized_data = torch.zeros((0)).to(device=self.device)

        self.dataset_params = dataset_params
        self.preload = dataset_params.preload
        self.slice_length = dataset_params.slice_length
        self.sample_rate = dataset_params.sample_rate

        self.tokenizer = tokenizer
        self.slice_dataset = slice_dataset
        self.buffer_process_batch_size = buffer_process_batch_size
        self.codebook_size = codebook_size
        self.epoch_size = epoch_size
        self.index_series_length = index_series_length

        # Check if the data is loaded
        indices_path = os.path.join(dataset_params.data_dir, "token_indices")
        if os.path.exists(indices_path):
            self._load_data(dataset_params.data_dir)
        else:
            if not os.path.exists(dataset_params.data_dir):
                os.makedirs(dataset_params.data_dir)
            self._generate_data()
            self._dump_data(dataset_params.data_dir)

    def _generate_data(self) -> None:
        """
        Generates tokenized data for the dataset.

        Raises:
            ValueError: If slice_dataset or tokenizer is not provided.
        """
        if self.slice_dataset is None:
            raise ValueError("Slice dataset must be provided if dataset doesn't exist")

        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer model must be provided if dataset doesn't exist"
            )

        track_filenames = list(set(self.slice_dataset.buffer["track_name"]))

        for track_filename in tqdm.tqdm(
            track_filenames, desc=f"Processing slice files..."
        ):
            index_track = self._process_single_track(track_filename)
            self.tokenized_data = torch.cat([self.tokenized_data, index_track], dim=0)

    def _process_single_track(self, track_name: str) -> torch.Tensor:
        """
        Process a single track and return the corresponding index tensor.

        Args:
            track_name (str): The name of the track to process.

        Returns:
            torch.Tensor: The index tensor representing the processed track.
        """
        if self.slice_dataset is None:
            raise ValueError("Slice dataset must be provided if dataset doesn't exist")

        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer model must be provided if dataset doesn't exist"
            )

        slice_data = torch.stack(
            [
                ind_slice_data
                for (ind_slice_data, ind_track_name) in zip(
                    self.slice_dataset.buffer["slice"],
                    self.slice_dataset.buffer["track_name"],
                )
                if ind_track_name == track_name
            ],
            dim=0,
        )

        slice_data_dataset = TensorDataset(slice_data)
        slice_data_loader = DataLoader(
            slice_data_dataset, batch_size=self.buffer_process_batch_size
        )

        index_track = torch.tensor((self.codebook_size)).unsqueeze(0).to(self.device)
        for slice_data_batch in tqdm.tqdm(
            slice_data_loader,
            desc=f"Processing slice batches for file {track_name}...",
        ):
            slice_data_batch = slice_data_batch[0].to(self.device)
            with torch.no_grad():
                tokenized_slice = self.tokenizer.model.tokenize(slice_data_batch)
            index_track = torch.cat([index_track, tokenized_slice.flatten()], dim=0)
        return torch.cat(
            [
                index_track,
                torch.tensor(self.codebook_size + 1)
                .unsqueeze(0)
                .to(index_track.device),
            ],
            dim=0,
        )

    def _dump_data(self, path: str) -> None:
        """
        Dump the tokenized data to the specified path.

        Args:
            path (str): The path to save the tokenized data.

        Returns:
            None
        """
        # Make slice folder
        tokenized_data_path = os.path.join(path, "token_indices")
        if not os.path.exists(tokenized_data_path):
            os.makedirs(tokenized_data_path)

        # Save the data
        torch.save(
            self.tokenized_data, os.path.join(tokenized_data_path, "tokenized_data.pt")
        )

    def _load_data(self, path: str) -> None:
        """
        Load tokenized data from the specified path.

        Args:
            path (str): The path to the directory containing the tokenized data.

        Returns:
            None
        """
        tokenized_data_path = os.path.join(path, "token_indices")
        self.tokenized_data = torch.load(
            os.path.join(tokenized_data_path, "tokenized_data.pt")
        ).to(self.device)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return self.tokenized_data.shape[0] - 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "indices": self._get_tokenized_data_slice_by_index(index).int(),
            "target": self._get_tokenized_data_slice_by_index(index + 1).long(),
        }

    def _get_tokenized_data_slice_by_index(self, index: int) -> torch.Tensor:
        if index + self.index_series_length > self.tokenized_data.shape[0]:
            tokenized_data_slice = torch.ones(self.index_series_length).to(
                device=self.device
            ) * (self.codebook_size + 1)
            tokenized_data_slice[: self.tokenized_data.shape[0] - index] = (
                self.tokenized_data[index : self.tokenized_data.shape[0]]
            )
        else:
            tokenized_data_slice = self.tokenized_data[
                index : index + self.index_series_length
            ]

        # Find the ending token in the slice if exists
        end_token = torch.where(tokenized_data_slice == self.codebook_size + 1)[0]
        if len(end_token) > 0:
            first_end_index = end_token[0]
            ones_slice = torch.ones_like(tokenized_data_slice).to(self.device) * (
                self.codebook_size + 1
            )
            ones_slice[:first_end_index] = tokenized_data_slice[:first_end_index]
            tokenized_data_slice = ones_slice

        return tokenized_data_slice

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> MP3TokenizedIndicesDataset:
        """
        Utility method to parse dataset parameters from a configuration dictionary.
        NOTE: This method is used to create an instance that only loads the dataset without creating it.

        Args:
            cfg (dict): The configuration dictionary.

        Returns:
            MP3TokenizedIndicesDataset: An instance of the class.
        """
        dataset_params = MusicDatasetParameters.from_cfg(cfg)
        codebook_size = cfg.model.vocabulary_size
        index_series_length = cfg.dataset.index_series_length
        slice_dataset = None
        tokenizer = None
        buffer_process_batch_size = 32
        epoch_size = cfg.dataset.epoch_size
        return cls(
            dataset_params,
            codebook_size,
            index_series_length,
            slice_dataset,
            tokenizer,
            buffer_process_batch_size,
            epoch_size,
        )

    def _right_pad_if_necessary(
        self, signal: torch.Tensor, signal_length: int
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
        if length_signal % signal_length != 0:
            num_missing_samples = signal_length - length_signal % signal_length
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding, value=self.codebook_size + 2)
        return signal
