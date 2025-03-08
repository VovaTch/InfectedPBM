import torch
import torch.nn.functional as F

from utils.waveform_tokenization import dequantize_waveform_256
from .quantized import QuantizedUint8MusicDataset


class MP3LongDataset(QuantizedUint8MusicDataset):
    """
    Dataset that returns the -1 to 1 soundwaves from the quantized dataset, both are build
    to train a possibly better tokenizer model.
    """

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        raw_data = super().__getitem__(idx)
        waveform = raw_data["slice"]
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform should be a tensor")
        raw_data["slice"] = dequantize_waveform_256(waveform).unsqueeze(0)
        return raw_data


class MP3VariableLongDataset(QuantizedUint8MusicDataset):
    """
    Dataset that returns the -1 to 1 soundwaves from the quantized dataset, both are build
    to train a possibly better tokenizer model. This one randomizes the length of the slice
    and pads the rest with 0 to account for variable slice length.
    """

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        raw_data = super().__getitem__(idx)
        waveform = raw_data["slice"]
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform should be a tensor")
        slice_length = torch.randint(1, self._slice_length, ()).item()
        raw_data["slice"] = dequantize_waveform_256(waveform[:slice_length])
        raw_data["slice"] = F.pad(
            raw_data["slice"], (0, self._slice_length - int(slice_length))
        ).unsqueeze(0)
        return raw_data
