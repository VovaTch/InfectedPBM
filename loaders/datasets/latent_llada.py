from typing import Any

import torch
from .latent import LatentSliceDataset


class LatentLladaSliceDataset(LatentSliceDataset):
    """
    A variation over the latent slice dataset that provides data for Large Language Diffusion Model's style
    training.
    """

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Retrieves a data point and applies a mask and conditional tensor.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            dict[str, Any]: A dictionary containing the data point with additional keys:
                - "mask": A tensor representing the mask applied to the latent data.
                - "conditional": A tensor representing the conditional value based on the slice path.
        """
        latent_slice_data_point = super().__getitem__(index)
        prob_to_mask = torch.rand((1,)).to(self._device)
        p_mask = (
            torch.ones((latent_slice_data_point["latent"].shape[0])).to(self._device)
            * prob_to_mask
        )
        mask = torch.bernoulli(p_mask).bool().to(self._device)
        conditional = self._track_name_to_idx_mapping.get(
            latent_slice_data_point["slice_path"],
            len(self._track_name_to_idx_mapping),
        )
        return {**latent_slice_data_point, "mask": mask, "conditional": conditional}
