import argparse
import hydra
from omegaconf import DictConfig

from common import logger, registry
from utils.containers import LearningParameters
from utils.trainer import initialize_trainer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script for InfectedBPM")
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to load the model from",
    )
    parser.add_argument(
        "-d",
        "--num_devices",
        type=int,
        default=1,
        help="Number of CUDA devices. If 0, use CPU.",
    )
    return parser.parse_args()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    args = get_args()
    learning_parameters = LearningParameters.from_cfg(cfg)
    cfg.learning.num_devices = args.num_devices

    # Data
    logger.info("Initializing data...")
    data_module = registry.get_data_module(cfg.dataset.data_module_type).from_cfg(cfg)

    # Trainer
    logger.info("Initializing trainer...")
    trainer = initialize_trainer(learning_parameters)

    # Initialize model
    logger.info("Initializing model...")
    if args.resume is not None:
        model = (
            registry.get_lightning_module(cfg.model.module_type)
            .from_cfg(cfg)
            .load_from_checkpoint(args.resume)
        )
    else:
        model = registry.get_lightning_module(cfg.model.module_type).from_cfg(cfg)

    # Train
    logger.info("Starting training...")
    trainer.test(model, data_module)
    logger.info("Finishing training.")


if __name__ == "__main__":
    main()
