from enum import Enum


class Stage(Enum):
    """
    An enum representing different stages in a typical machine learning pipeline.

    - `TRAIN`: Used for the training phase where model parameters are learned.
    - `VALIDATION`: Used for the validation phase to tune hyperparameters and evaluate the model's performance.
    - `TEST`: Used for the testing phase to evaluate the model's performance on unseen data.
    """

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
