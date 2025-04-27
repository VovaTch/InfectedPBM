import argparse
import os

import torch
import torchaudio
import tqdm

from utils.logger import logger


def main(args: argparse.Namespace) -> None:

    os.makedirs(args.output_dir, exist_ok=True)
    for idx in tqdm.tqdm(range(args.num_waveforms), desc="Creating waveforms..."):
        waveform = torch.rand(int(args.length * args.sample_rate)) * 2 - 1
        waveform = waveform.unsqueeze(0)
        output_path = os.path.join(args.output_dir, f"test_sound_{idx + 1}.mp3")
        torchaudio.save(output_path, waveform, args.sample_rate, format="mp3")
        logger.info(f"Saved test sound to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test sounds")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=os.path.join("tests", "data"),
        help="Output folder path",
    )
    parser.add_argument(
        "-s", "--sample_rate", type=int, default=44100, help="Sample rate"
    )
    parser.add_argument(
        "-l", "--length", type=float, default=2.0, help="Length in seconds"
    )
    parser.add_argument(
        "-n", "--num_waveforms", type=int, default=3, help="Number of waveforms"
    )
    args = parser.parse_args()
    main(args)
