import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm

from models.autoregressor_module import AutoRegressorModule
from models.base import Tokenizer


class AutoRegressivePipeline:
    """
    Pipeline object for utilizing the auto-regressive model for music, a simplified version that generates a sequence
    of predicted tokens given an input sequence, temperature, and top-k sampling.
    """

    def __init__(self, tokenizer: Tokenizer, ar_predictor: AutoRegressorModule) -> None:
        """
        Initialize the Pipeline object.

        Args:
            tokenizer (Tokenizer): The tokenizer object used for tokenizing input data.
            ar_predictor (AutoRegressorModule): The auto-regressor module used for prediction.
        """
        self._tokenizer = tokenizer
        self._ar_predictor = ar_predictor

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
        with torch.no_grad():
            last_pred_logits: torch.Tensor = self._ar_predictor.model.get_last_logits(
                index_sequence
            )  # BS x L x num_CB

            # Temperature
            last_pred_logits /= float(temperature)

            # Top-k
            if top_k > 0:
                values, indices = torch.topk(
                    last_pred_logits.transpose(1, 2), top_k, dim=2
                )
                dist = torch.distributions.Categorical(logits=values)
                sample = dist.sample().unsqueeze(1)
                # sampled_tokens = indices.gather(1, sample.unsqueeze(1)).squeeze(1)
                sampled_tokens = torch.gather(
                    indices, 2, sample.transpose(1, 2)
                ).transpose(1, 2)
            else:
                dist = torch.distributions.Categorical(
                    logits=last_pred_logits.transpose(1, 2)
                )
                sampled_tokens = dist.sample().unsqueeze(-1).transpose(1, 2)

            return sampled_tokens  # BS x 1 x num_CB

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
        with torch.no_grad():
            for _ in tqdm.tqdm(range(length), desc="Generating tokens..."):
                next_token = self.predict_next_token(
                    index_sequence, temperature=temperature, top_k=top_k
                )
                index_sequence = torch.cat([index_sequence, next_token], dim=1)

            return index_sequence

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
        start_index = self._ar_predictor.model.vocabulary_size
        end_index = self._ar_predictor.model.vocabulary_size + 1
        index_sequence = self.predict_fixed_token_series(
            index_sequence, length, temperature, top_k
        )

        # Filter out start and end tokens
        mask = torch.isin(
            index_sequence,
            torch.tensor([start_index, end_index]).to(index_sequence.device),
        ).any(dim=-1)
        index_sequence_filtered = index_sequence[:, ~mask.squeeze(0)]

        index_sequence_iterator = DataLoader(
            TensorDataset(index_sequence_filtered.flatten(end_dim=-2).unsqueeze(-1)),
            batch_size=translation_batch_size,
        )

        for index_sequence_batch in tqdm.tqdm(
            index_sequence_iterator, desc="Creating sound-wave..."
        ):
            translated_batch = self._tokenizer.from_tokens(index_sequence_batch[0])
            if not hasattr(self, "translated_sequence"):
                translated_sequence = translated_batch.view(1, -1).contiguous()
            else:
                translated_sequence = torch.cat(
                    (translated_sequence, translated_batch), dim=-1
                )
        return translated_sequence
