import os
import subprocess

import hydra
import fire

from common import logger
from utils.config import load_cfg_from_hydra


def main(checkpoint_name: str, folder_path: str = "weights") -> None:
    """
    Copy the last checkpoint file to a new location and delete the old checkpoint file.

    Args:
        checkpoint_name (str): The name of the new checkpoint file.
        folder_path (str, optional): The path to the folder where the new checkpoint file will be saved. Defaults to "weights".
    """
    cfg = load_cfg_from_hydra("../config", "config")

    # Create the folder if there is none
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create paths
    last_checkpoint_save_path = os.path.join(
        cfg.learning.save_path, cfg.model_name, "last.ckpt"
    )
    new_saved_path = os.path.join(folder_path, f"{checkpoint_name}.ckpt")
    logger.info(f"Copying {last_checkpoint_save_path} to {new_saved_path}")

    # Create and run the new command
    command = ["cp", last_checkpoint_save_path, new_saved_path]
    subprocess.run(command)
    logger.info("Copied the last checkpoint to the new path.")

    # delete the old last checkpoint
    command = ["rm", last_checkpoint_save_path]
    subprocess.run(command)
    logger.info("Deleted the old last checkpoint.")


if __name__ == "__main__":
    fire.Fire(main)
