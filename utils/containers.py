from dataclasses import dataclass
from typing import Any, Type, TypeVar

from omegaconf import DictConfig


_T = TypeVar("_T")


@dataclass
class LearningParameters:

    """
    Class representing learning parameters for a model.

    Attributes:
        model_name (str): Name of the model.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        beta_ema (float): Beta value for exponential moving average.
        gradient_clip (float): Value for gradient clipping.
        save_path (str): Path to save the trained model.
        eval_split_factor (float): Split factor for evaluation data.
        amp (bool): Flag indicating whether to use automatic mixed precision.
        val_split (float): Split factor for validation data.
        test_split (float): Split factor for test data, defaults to 0.0.
        num_devices (int, optional): Number of devices to use for training. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        loss_monitor (str, optional): Loss monitor for scheduler. Defaults to "step".
        interval (str, optional): Interval for scheduler. Defaults to "training_total_loss".
        frequency (int, optional): Frequency for scheduler. Defaults to 1.
        use_wandb (bool, optional): Flag indicating whether to use wandb. Defaults to False.
        project_name (str): Project name, used mostly for Wandb, defaults to InfectedBPM.
    """

    model_name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    beta_ema: float
    gradient_clip: float
    save_path: str
    amp: bool
    val_split: float
    use_wandb: bool = False
    project_name: str = "InfectedBPM"
    test_split: float = 0.0
    num_devices: int = 1
    num_workers: int = 0
    loss_monitor: str = "step"
    interval: str = "training_total_loss"
    frequency: int = 1

    @classmethod
    def from_cfg(cls: type[_T], cfg: DictConfig) -> _T:
        """
        Utility method to parse learning parameters from a configuration dictionary

        Args:
            cfg (DictConfig): configuration dictionary

        Returns:
            LearningParameters: Learning parameters object
        """
        learning_params = cfg.learning
        return cls(
            model_name=cfg["model_name"],
            learning_rate=learning_params["learning_rate"],
            weight_decay=learning_params["weight_decay"],
            batch_size=learning_params["batch_size"],
            epochs=learning_params["epochs"],
            beta_ema=learning_params["beta_ema"],
            gradient_clip=learning_params["gradient_clip"],
            save_path=learning_params["save_path"],
            val_split=learning_params["val_split"],
            test_split=learning_params["test_split"],
            amp=learning_params["amp"],
            loss_monitor=learning_params["loss_monitor"],
            interval=learning_params["interval"],
            frequency=learning_params["frequency"],
            project_name=cfg["project_name"],
            use_wandb=cfg["use_wandb"],
            # num_workers=learning_params["num_workers"],
            # num_devices=learning_params["num_devices"],
        )


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

    @classmethod
    def from_cfg(cls: type[_T], cfg: dict[str, Any]) -> _T:
        """
        Utility method to parse Mel spectrogram parameters from a configuration dictionary

        Args:
            cfg (DictConfig): configuration dictionary

        Returns:
            MelSpecParameters: Mel spectrogram parameters object
        """
        return cls(
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
            power=cfg["power"],
            f_min=cfg["f_min"],
            pad=cfg["pad"],
            pad_mode=cfg["pad_mode"],
            norm=cfg["norm"],
            mel_scale=cfg["mel_scale"],
        )


@dataclass
class MusicDatasetParameters:
    """
    Parameters for the MusicDataset class.

    Attributes:
        dataset_type (str): The type of the dataset.
        data_module_type (str): The type of the data module.
        sample_rate (int): The sample rate of the audio data.
        data_dir (str): The directory containing the data.
        slice_length (int): The length of each audio slice.
        preload (bool): Whether to preload the data.
        preload_data_dir (str): The directory for preloaded data.
        device (str, optional): The device to use (default is "cpu").
    """

    dataset_type: str
    data_module_type: str
    sample_rate: int
    data_dir: str
    slice_length: int
    preload: bool
    preload_data_dir: str
    device: str = "cpu"

    @classmethod
    def from_cfg(cls: type[_T], cfg: DictConfig) -> _T:
        """
        Utility method to parse dataset parameters from a configuration dictionary

        Args:
            cfg (DictConfig): configuration dictionary

        Returns:
            MusicDatasetParameters: dataset parameters object
        """
        dataset_cfg = cfg.dataset
        return cls(
            dataset_type=dataset_cfg["dataset_type"],
            data_module_type=dataset_cfg["data_module_type"],
            sample_rate=dataset_cfg["sample_rate"],
            data_dir=dataset_cfg["data_dir"],
            slice_length=dataset_cfg["slice_length"],
            preload=dataset_cfg["preload"],
            preload_data_dir=dataset_cfg["preload_data_dir"],
            device=dataset_cfg["device"],
        )
