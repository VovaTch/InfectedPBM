from turtle import forward
from typing import Any

import torch
from loaders.datasets.latent import Tokenizer
from .latent import LatentSliceDataset


class LatentLladaSliceDataset(LatentSliceDataset):
    """
    A variation over the latent slice dataset that provides data for Large Language Diffusion Model's style
    training.
    """

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
        super().__init__(
            data_path,
            slices_per_sample,
            tokenizer,
            tokenizing_batch_size,
            device,
            tokenizing_device,
            channel_first_data,
            sample_rate,
            processed_path,
            save_processed,
            slice_level,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        latent_slice_data_point = super().__getitem__(index)
        prob_to_mask = torch.rand().to(self._device)
        p_mask = (
            torch.ones_like(latent_slice_data_point["latent_slice"]).to(self._device)
            * prob_to_mask
        )
        mask = torch.bernoulli(p_mask).to(self._device)
        return {**latent_slice_data_point, "mask": mask}
