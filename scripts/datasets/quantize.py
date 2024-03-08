import os

import torch
import torchaudio
import hydra
from omegaconf import DictConfig
import tqdm

from utils.waveform_tokenization import (
    encode_waveform_256,
    merge_and_tokenize_256,
    quantize_waveform_256,
)


VOCABULARY_SIZE = 276


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Collect all the data into a single looooong tensor
    long_waveform = []
    mp3_path = os.path.join(cfg.dataset.data_dir, "minimal_testing")
    for file in tqdm.tqdm(os.listdir(mp3_path), desc="Processing MP3 files..."):
        if file.endswith(".mp3"):
            waveform, _ = torchaudio.load(os.path.join(mp3_path, file))  # type: ignore
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform_quantized = quantize_waveform_256(waveform)
            long_waveform += waveform_quantized.flatten().tolist()

    print(f"Waveform length is {len(long_waveform):,}")

    # Try to tokenize to reduce the length
    num_merges = VOCABULARY_SIZE - 256
    encoded_waveform, merges = merge_and_tokenize_256(long_waveform, num_merges)
    # TODO: save merges

    print(f"Encoded waveform length is {len(encoded_waveform):,}")


if __name__ == "__main__":
    main()
