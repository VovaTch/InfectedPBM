from torch._tensor import Tensor

from utils.waveform_tokenization import dequantize_waveform_256
from .quantized import QuantizedUint8MusicDataset


class MP3LongDataset(QuantizedUint8MusicDataset):
    """
    Dataset that returns the -1 to 1 soundwaves from the quantized dataset, both are build
    to train a possibly better tokenizer model.
    """

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        raw_data = super().__getitem__(idx)
        waveform = raw_data["waveform"]
        if not isinstance(waveform, Tensor):
            raise TypeError("waveform should be a tensor")
        raw_data["waveform"] = dequantize_waveform_256(waveform)
        return raw_data
