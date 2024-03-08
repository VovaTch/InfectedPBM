from typing import Any

from torch.utils.data import Dataset

from common import registry
from utils.containers import MusicDatasetParameters


@registry.register_dataset("quantized_uint8")
class QuantizedUint8MusicDataset(Dataset):
    """
    Dataset that takes soundwaves as torch, and quantizes them to uint8. This dataset
    will be used to train a tokenizer that works similarly to a language model. 8-bit
    quantized IM tracks sounds exactly like the MP3 versions (the MP3 version might be
    quantized already so whatever...)
    """

    dataset_params: MusicDatasetParameters
    buffer: dict[str, Any] = {}

    def __init__(self, dataset_params: MusicDatasetParameters) -> None:
        super().__init__()

        self.dataset_params = dataset_params
        self.preload = dataset_params.preload
