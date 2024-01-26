import argparse

import hydra
from omegaconf import DictConfig

from utils.containers import LearningParameters
from utils.trainer import initialize_trainer
from common import registry, logger


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training script for the MNIST template"
    )
    parser.add_argument(
        "-d",
        "--num_devices",
        type=int,
        default=1,
        help="Number of CUDA devices. If 0, use CPU.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="mnist_fcn",
        help="Type of model used for training.",
    )
    parser.add_argument(
        "-dm",
        "--data_module",
        type=str,
        default="mnist",
        help="Type of data module used for training",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to load the model from",
    )
    return parser.parse_args()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    args = get_args()

    cfg.learning.num_devices = args.num_devices
    learning_params = LearningParameters.from_cfg(cfg)

    logger.debug("Loading trainer...")
    trainer = initialize_trainer(learning_params)

    logger.debug("Loading model...")
    model = registry.get_lightning_module(args.model)(cfg, args.resume)

    logger.debug("Loading data module...")
    data_module = registry.get_data_module(args.data_module).from_cfg(cfg)  # type: ignore

    logger.info("Initiating testing...")
    trainer.test(model, data_module)
    logger.info("Finished testing")


if __name__ == "__main__":
    main()
