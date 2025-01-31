import hydra
from matplotlib import pyplot as plt
from numpy import single
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset

from utils.containers import LearningParameters
from common import registry, logger
from utils.trainer import initialize_trainer


class SingleSampleDataset(Dataset):
    def __init__(
        self, original_dataset: Dataset, sample_idx: int, length: int = 1
    ) -> None:
        self.original_dataset = original_dataset
        self.sample_idx = sample_idx
        self.original_data = original_dataset[sample_idx : sample_idx + length]
        self.original_data["slice"] = torch.concat(self.original_data["slice"], dim=1)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.original_data


@hydra.main(version_base=None, config_path="../config", config_name="ripl")
def main(cfg: DictConfig) -> None:
    learning_parameters = LearningParameters.from_cfg(cfg)

    # Data
    dataset = registry.get_dataset(cfg.dataset.dataset_type).from_cfg(cfg)
    single_sample_dataset = SingleSampleDataset(dataset, 5500, 1)  # type: ignore

    # Loader
    loader = DataLoader(single_sample_dataset, batch_size=1, shuffle=False)

    # Model
    model = registry.get_lightning_module(cfg.model.module_type).from_cfg(
        cfg, cfg.resume
    )

    # Trainer
    trainer = initialize_trainer(learning_parameters)
    trainer.fit(model, loader)

    plt.plot(single_sample_dataset[0]["slice"].squeeze().cpu().numpy())
    sample_slice = {"slice": single_sample_dataset[0]["slice"].unsqueeze(0).to("cpu")}
    print(sample_slice["slice"].shape)
    sample_slice = {"slice": torch.zeros((1, 1, 32768), dtype=torch.float32).to("cpu")}
    plt.plot(model(sample_slice)["slice"].squeeze().detach().cpu().numpy())
    plt.show()


if __name__ == "__main__":
    main()
