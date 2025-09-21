from dataclasses import dataclass
from typing import Any


@dataclass
class LearningParameters:
    """
    Learning parameters dataclass to contain every parameter required for training,
    excluding the optimizer and the scheduler, which are handled separately.
    """

    model_name: str
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    batch_size: int = 32
    grad_accumulation: int = 1
    epochs: int = 10
    beta_ema: float = 0.999
    gradient_clip: float | None = 0.5
    save_path: str = "saved"
    amp: bool = False
    val_split: float = 0.05
    test_split: float = 0.01
    devices: Any = "auto"
    num_workers: int = 0
    loss_monitor: str = "validation total loss"
    trigger_loss: float = 0.0
    interval: str = "step"
    frequency: int = 1
    limit_train_batches: int | float | None = None
    limit_eval_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    pin_memory: bool = True


@dataclass
class MelSpecParameters:
    """
    A class representing the parameters for computing Mel spectrograms.

    Attributes:
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples between successive frames.
        n_mels (int): The number of Mel frequency bins.
        power (float): The exponent for the magnitude spectrogram.
        f_min (float): The minimum frequency of the Mel filter bank.
        pad (int): The number of padding points.
        pad_mode (str, optional): The padding mode. Defaults to "reflect".
        norm (str, optional): The normalization mode. Defaults to "slaney".
        mel_scale (str, optional): The Mel scale type. Defaults to "htk".
    """

    n_fft: int
    hop_length: int
    n_mels: int
    power: float
    f_min: float
    pad: int
    pad_mode: str = "reflect"
    norm: str = "slaney"
    mel_scale: str = "htk"
