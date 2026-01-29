import warnings

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    ModelSummary,
)
from lightning.pytorch.loggers import Logger, TensorBoardLogger

from .ema import EMA
from .containers import LearningParameters


def get_trainer(learning_parameters: LearningParameters) -> L.Trainer:
    """
    Initializes a Pytorch Lightning training, given a learning parameters object

    Args:
        learning_parameters (LearningParameters): learning parameters object

    Returns:
        pl.Trainer: Pytorch lightning trainer
    """
    # Set device
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available, using CPU")
        devices = "auto"
        accelerator = "cpu"
    else:
        devices = learning_parameters.devices
        accelerator = (
            "cpu" if learning_parameters.devices == "cpu" or devices == "cpu" else "gpu"
        )

    save_folder = learning_parameters.save_path

    # Configure trainer
    ema = EMA(learning_parameters.beta_ema)
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(
        save_dir=save_folder, name=learning_parameters.model_name
    )
    loggers: list[Logger] = [tensorboard_logger]

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=save_folder,
        filename=learning_parameters.model_name,
        save_weights_only=True,
        save_top_k=1,
        monitor=learning_parameters.loss_monitor,
        enable_version_counter=False,
    )
    early_stopping = EarlyStopping(
        monitor=learning_parameters.loss_monitor,
        stopping_threshold=learning_parameters.trigger_loss,
        patience=int(
            learning_parameters.epochs
        ),  # Early stopping is here is for only stopping once the training reached a threshold loss.
    )

    # AMP
    precision = 16 if learning_parameters.amp else 32

    model_summary = ModelSummary(max_depth=2)
    trainer = L.Trainer(
        gradient_clip_val=learning_parameters.gradient_clip,
        logger=loggers,
        callbacks=[
            early_stopping,
            model_checkpoint_callback,
            model_summary,
            learning_rate_monitor,
            ema,
        ],
        strategy="ddp",
        devices=devices,
        max_epochs=learning_parameters.epochs,
        log_every_n_steps=1,
        precision=precision,
        accelerator=accelerator,
        accumulate_grad_batches=learning_parameters.grad_accumulation,
        limit_train_batches=learning_parameters.limit_train_batches,
        limit_val_batches=learning_parameters.limit_eval_batches,
        limit_test_batches=learning_parameters.limit_test_batches,
    )

    return trainer
