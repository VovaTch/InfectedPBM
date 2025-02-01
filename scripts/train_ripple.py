import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset

from utils.logger import logger
from utils.learning import get_trainer


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

    # Setting precision...
    torch.set_float32_matmul_precision("high")

    # Data
    dataset = hydra.utils.instantiate(cfg.data.dataset)
    single_sample_dataset = SingleSampleDataset(dataset, 5500, 1)  # type: ignore

    # Loader
    loader = DataLoader(single_sample_dataset, batch_size=1, shuffle=False)

    # Model
    model = hydra.utils.instantiate(cfg.model)

    # Trainer
    learning_parameters = hydra.utils.instantiate(cfg.learning)
    trainer = get_trainer(learning_parameters)
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, loader)
    logger.info("Finishing training...")

    plt.plot(single_sample_dataset[0]["slice"].squeeze().cpu().numpy())
    sample_slice = {"slice": single_sample_dataset[0]["slice"].unsqueeze(0).to("cpu")}
    print(sample_slice["slice"].shape)
    sample_slice = {"slice": torch.zeros((1, 1, 32768), dtype=torch.float32).to("cpu")}
    plt.plot(model(sample_slice)["slice"].squeeze().detach().cpu().numpy())
    plt.show()


if __name__ == "__main__":
    main()
