import matplotlib.pyplot as plt
import torch

from utils.sample_schedulers.linear import (
    LinearEntropyBatchSampleScheduler,
    LinearSampleScheduler,
)


def main() -> None:
    init_mask = torch.zeros((256)).bool()
    batch_len = 32
    steps_per_batch = 100
    scheduler = LinearEntropyBatchSampleScheduler(batch_len, steps_per_batch)
    logits = torch.randn((256, 8))

    masks = [init_mask]
    for step in range((256 // batch_len) * steps_per_batch):
        mask = scheduler.sample(step, masks[-1], logits)
        masks.append(mask)
    total_masks = torch.stack(masks).float()
    plt.matshow(total_masks.int().detach().cpu().numpy(), aspect="auto", vmin=0, vmax=1)
    plt.show()

    init_mask = torch.zeros((256)).bool()
    steps = 100
    scheduler = LinearSampleScheduler(steps)
    masks = [init_mask]
    for step in range(steps):
        mask = scheduler.sample(step, masks[-1], logits)
        masks.append(mask)
    total_masks = torch.stack(masks).float()
    plt.matshow(total_masks.int().detach().cpu().numpy(), aspect="auto", vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    main()
