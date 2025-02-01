import hydra
from omegaconf import DictConfig
import torch

from models.modules import load_inner_model_state_dict
from utils.logger import logger
from utils.learning import get_trainer


@hydra.main(version_base=None, config_path="../config", config_name="lvl1_vqvae")
def main(cfg: DictConfig) -> None:
    """
    Main function for training. Initializes the data, model, and trainer and starts the training.
    """

    # Setting precision...
    torch.set_float32_matmul_precision("high")

    # Data
    logger.info("Initializing data...")
    data_module = hydra.utils.instantiate(cfg.data)

    # Trainer
    logger.info("Initializing trainer...")
    learning_params = hydra.utils.instantiate(cfg.learning)
    trainer = get_trainer(learning_params)

    # Initialize model
    module = hydra.utils.instantiate(cfg.module, _convert_="partial").to("cuda")
    if cfg.use_torch_compile:
        module.model = torch.compile(module.model)
    if cfg.resume is not None:
        module = load_inner_model_state_dict(module, cfg.resume)

    # Train
    logger.info("Starting training...")
    trainer.fit(module, data_module)
    logger.info("Finishing training.")


if __name__ == "__main__":
    main()
