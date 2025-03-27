from __future__ import annotations
from enum import Enum, auto
import json
import os
from typing import Any

import torch
from torch.utils.data import Dataset
import tqdm

from utils.logger import logger
from .latent import LatentDataPoint


class LatentLevel(Enum):
    LOW = auto()
    HIGH = auto()


class DualLatentDataset(Dataset):
    buffer: list[LatentDataPoint]

    def __init__(
        self,
        low_lvl_data_path: str,
        high_lvl_data_path: str,
        sample_rate: int = 44100,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._low_lvl_data_path = low_lvl_data_path
        self._high_lvl_data_path = high_lvl_data_path
        self._sample_rate = sample_rate
        self._device = device

        logger.info(f"Loading low-level data from {low_lvl_data_path}")
        low_lvl_data = self._load_data(low_lvl_data_path, LatentLevel.LOW)
        logger.info(f"Loading high-level data from {high_lvl_data_path}")
        high_lvl_data = self._load_data(high_lvl_data_path, LatentLevel.HIGH)

        logger.info("Merging low-level and high-level data")
        self._buffer = self._merge_data(low_lvl_data, high_lvl_data)
        logger.info("Data loaded successfully")

    def _load_data(
        self, data_path: str, latent_level: LatentLevel
    ) -> list[LatentDataPoint]:

        load_message = {
            LatentLevel.LOW: "low level",
            LatentLevel.HIGH: "high level",
        }

        loaded_data = []

        json_file_path = os.path.join(data_path, "_metadata.json")
        with open(json_file_path, "r") as f:
            metadata: list[dict[str, Any]] = json.load(f)

        latent_files: list[str] = list(
            sorted({data_point["latent_path"] for data_point in metadata})
        )
        for latent_file in tqdm.tqdm(
            latent_files, f"Loading {load_message[latent_level]} latents..."
        ):
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

        return loaded_data
