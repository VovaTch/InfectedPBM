import hydra
from omegaconf import DictConfig
import torch
from loaders.datasets.indices import MP3TokenizedIndicesDataset

from utils.containers import MusicDatasetParameters
from common import registry, logger


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:

    dataset_parameters = MusicDatasetParameters.from_cfg(cfg)
    dataset_parameters.device = "cuda" if torch.cuda.is_available() else "cpu"
    slice_dataset = registry.get_dataset(cfg.dataset.dataset_type).from_cfg(cfg)
    tokenizer = (
        registry.get_lightning_module(cfg.model.module_type)
        .from_cfg(cfg)
        .to(dataset_parameters.device)
    )

    tokenized_dataset = MP3TokenizedIndicesDataset(
        dataset_parameters,
        cfg.model.vocabulary_size,
        slice_dataset=slice_dataset,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
    )

    logger.info(
        f"Tokenized dataset size: {len(tokenized_dataset)}, "
        f"tokenized data sample shape: {tokenized_dataset[0]['indices'].shape}"
    )


if __name__ == "__main__":
    main()
