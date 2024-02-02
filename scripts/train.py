import hydra
from omegaconf import DictConfig

from common import logger, registry
from utils.containers import LearningParameters
from utils.trainer import initialize_trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    learning_parameters = LearningParameters.from_cfg(cfg)

    # Data
    logger.info("Initializing data...")
    data_module = registry.get_data_module(cfg.dataset.data_module_type).from_cfg(cfg)

    # Trainer
    logger.info("Initializing trainer...")
    trainer = initialize_trainer(learning_parameters)

    # Initialize model
    logger.info("Initializing model...")
    model = registry.get_lightning_module(cfg.model.module_type).from_cfg(
        cfg, cfg.resume
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    logger.info("Finishing training.")


if __name__ == "__main__":
    main()
