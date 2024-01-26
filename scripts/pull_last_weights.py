import argparse
import os
from typing import Any
import subprocess


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to pull the last weights from a model"
    )
    parser.add_argument(
        "-p", "--path", type=str, default="saved", help="Path for all the saved files"
    )
    parser.add_argument(
        "-n", "--model_name", type=str, default="mnist_fcn", help="Name of the model"
    )
    parser.add_argument(
        "-f",
        "--file_name",
        type=str,
        default="mnist_best",
        help="File name to copy the weights into",
    )
    parser.add_argument(
        "-del",
        "--delete",
        action="store_true",
        help="Delete history; default is False",
    )
    return parser.parse_args()


def _parse_args(args: Any) -> tuple[str, str, str, bool]:
    save_path = args.path
    model_name = args.model_name
    file_name = args.file_name
    delete_history = args.delete
    return save_path, model_name, file_name, delete_history


def list_direct_subdirs(root_dir_path: str) -> list[str]:
    """
    Given a path, list all subdirectories in that path

    Args:
        root_dir_path (str): Folder path

    Returns:
        list[str]: All subdirectories in the path
    """
    version_dirs = []
    for item in os.listdir(root_dir_path):
        if not os.path.isdir(os.path.join(root_dir_path, item)):
            continue
        version_dirs.append(item)
    return version_dirs


def find_latest_version_dir(version_dirs: list[str]) -> str:
    """
    Utility function to find the latest version folder out of a list of folders with identical
    names except for the version number

    Args:
        version_dirs (list[str]): List of version directories

    Returns:
        str: The last version directory
    """
    version_dirs_number = [int(ind_dir.split("_")[-1]) for ind_dir in version_dirs]
    max_version_idx = version_dirs_number.index(max(version_dirs_number))
    return version_dirs[max_version_idx]


def main() -> None:
    """
    Script to copy the latest version of a model into a "weights" folder.
    """

    args = get_args()

    save_path, model_name, file_name, delete_history = _parse_args(args)
    saved_versions = os.path.join(save_path, model_name)
    version_dirs = list_direct_subdirs(saved_versions)
    latest_version_dir = find_latest_version_dir(version_dirs)
    weights_file_path = os.path.join(
        saved_versions, latest_version_dir, "checkpoints", "last.ckpt"
    )
    if not os.path.isdir("weights"):
        os.makedirs("weights")
    moved_weights_file_path = os.path.join("weights", file_name + ".ckpt")

    # Create and execute copy shell command
    command = ["cp", "-r", weights_file_path, moved_weights_file_path]
    subprocess.run(command)
    print(f"Copied weights from {weights_file_path} to {moved_weights_file_path}")

    # Create and execute delete shell command if needed
    command = ["rm", "-r", saved_versions]
    if delete_history:
        subprocess.run(command)
        print(f"Removed all model checkpoints from {saved_versions}")


if __name__ == "__main__":
    main()
