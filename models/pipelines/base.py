from abc import ABC, abstractmethod

import torch


class Pipeline(ABC):
    @abstractmethod
    def predict_next_token(
        self, index_sequence: torch.Tensor, temperature: float = 1.0, top_k: int = -1
    ) -> torch.Tensor:
        """
        Predicts the next token in a sequence given an input index sequence.

        Args:
            index_sequence (torch.Tensor): The input index sequence.
            temperature (float, optional): The temperature parameter for controlling the randomness of the predictions.
                Higher values (e.g., > 1.0) make the predictions more random, while lower values (e.g., < 1.0) make them more deterministic.
                Defaults to 1.0.
            top_k (int, optional): The number of top-k tokens to consider during sampling.
                If set to a positive value, only the top-k tokens with the highest probabilities will be considered.
                If set to a negative value (default), all tokens will be considered.

        Returns:
            torch.Tensor: The predicted next token(s) in the sequence.
        """
        ...

    @abstractmethod
    def predict_fixed_token_series(
        self,
        index_sequence: torch.Tensor,
        length: int,
        temperature: float = 1.0,
        top_k: int = -1,
    ) -> torch.Tensor:
        """
        Predicts a fixed-length series of tokens given an input index sequence.

        Args:
            index_sequence (torch.Tensor): The input index sequence.
                A tensor representing the input index sequence.
            length (int): The length of the additional output token series.
                An integer representing the length of the additional output token series.
            temperature (float, optional): The temperature value for sampling.
                A float representing the temperature value for sampling. Default is 1.0.
            top_k (int, optional): The number of top-k tokens to consider for sampling.
                An integer representing the number of top-k tokens to consider for sampling. Default is -1.

        Returns:
            torch.Tensor: The predicted token series.
                A tensor representing the predicted token series.
        """
        ...

    @abstractmethod
    def create_fixed_music_slice(
        self,
        index_sequence: torch.Tensor,
        length: int,
        temperature: float = 1.0,
        top_k: int = -1,
        translation_batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Generates a fixed-length music slice based on the given index sequence.

        Args:
            index_sequence (torch.Tensor): The input index sequence, size BS x L x num_CB.
            length (int): The desired length of the generated music slice.
            temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
            top_k (int, optional): The number of top-k tokens to consider during sampling. Defaults to -1.
            translation_batch_size (int, optional): The batch size for translation. Defaults to 32.

        Returns:
            torch.Tensor: The generated music slice as a tensor.

        """
        ...
