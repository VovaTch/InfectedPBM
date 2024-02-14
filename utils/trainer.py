import os
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, Logger

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

    # Configure trainer
    ema = EMA(learning_parameters.beta_ema)
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(
        save_dir="saved/", name=learning_parameters.model_name
    )
    loggers: list[Logger] = [tensorboard_logger]

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

    # Initialize wandb if needed
    if learning_parameters.use_wandb:
        wandb_logger = WandbLogger(
            project=learning_parameters.project_name, log_model="all"
        )
        loggers.append(wandb_logger)

    # AMP
    precision = 16 if learning_parameters.amp else 32

    model_summary = ModelSummary(max_depth=3)
    trainer = L.Trainer(
        gradient_clip_val=learning_parameters.gradient_clip,
        logger=loggers,
        callbacks=[
            model_checkpoint_callback,
            model_last_checkpoint_callback,
            model_summary,
            learning_rate_monitor,
            ema,
        ],
        devices="auto",
        max_epochs=learning_parameters.epochs,
        log_every_n_steps=1,
        precision=precision,
        accelerator=accelerator,
    )

    return trainer
