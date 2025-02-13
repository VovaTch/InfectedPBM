from typing import Literal

import torch
import torch.nn as nn


class MixtureOfExpertsRotaryStftDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_experts: int,
        top_k_gating: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        max_seq_len: int = 32768,
        ff_hidden_dim: int = 2048,
        norm_type: Literal["rmsnorm", "layernorm"] = "layernorm",
        dropout: float = 0.1,
        padding: Literal["center", "same"] = "same",
    ) -> None:
        super().__init__()
