import torch
from torch.utils.data import Dataset


class RandomWaveformDataset(Dataset):
    def __init__(
        self,
        slice_length: int,
        input_channels: int = 1,
        output_class: int = 0,
        device: str = "cpu",
        dataset_length: int = 1000,
    ) -> None:
        super().__init__()

        self._slice_length = slice_length
        self._input_channels = input_channels
        self._device = device
        self._dataset_length = dataset_length
        self._output_class = output_class

    def __len__(self) -> int:
        return self._dataset_length

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        waveform = torch.randn(
            self._input_channels, self._slice_length, device=self._device
        )
        return {"slice": waveform, "label": self._output_class}
