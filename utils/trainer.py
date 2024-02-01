import os
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig

from .ema import EMA
from .containers import LearningParameters


def initialize_trainer(learning_parameters: LearningParameters) -> L.Trainer:
    """
    Initializes a Pytorch Lightning training, given a learning parameters object

    Args:
        learning_parameters (LearningParameters): learning parameters object

    Returns:
        pl.Trainer: Pytorch lightning trainer
    """
    # Set device
    num_devices = learning_parameters.num_devices
    accelerator = "cpu" if num_devices == 0 else "gpu"
    device_list = [idx for idx in range(num_devices)]

    # Configure trainer
    ema = EMA(learning_parameters.beta_ema)
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir="saved/", name=learning_parameters.model_name)
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("saved", learning_parameters.model_name),
        filename=f"{learning_parameters.model_name}_best.ckpt",
        save_weights_only=True,
        save_top_k=1,
        monitor=learning_parameters.loss_monitor,
    )
    model_last_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("saved", learning_parameters.model_name),
        filename=f"{learning_parameters.model_name}_last.ckpt",
        save_last=True,
        save_weights_only=True,
        save_top_k=0,
    )

    # AMP
    precision = 16 if learning_parameters.amp else 32

    model_summary = ModelSummary(max_depth=3)
    trainer = L.Trainer(
        gradient_clip_val=learning_parameters.gradient_clip,
        logger=logger,
        callbacks=[
            model_checkpoint_callback,
            model_last_checkpoint_callback,
            model_summary,
            learning_rate_monitor,
            ema,
        ],
        devices=device_list,
        max_epochs=learning_parameters.epochs,
        log_every_n_steps=1,
        precision=precision,
        accelerator=accelerator,
    )

    return trainer


def add_optional_wandb_logger(trainer: L.Trainer, cfg: DictConfig) -> None:
    """
    Adds a wandb logger to the trainer, if wandb is used

    Args:
        trainer (L.Trainer): Pytorch lightning trainer
        cfg (DictConfig): Hydra config
    """
    if cfg.use_wandb:
        wandb_logger = WandbLogger(project=cfg.project_name, log_model="all")
        trainer.loggers = [*trainer.loggers, wandb_logger]
