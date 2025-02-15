from curses import window
from typing import Literal

import torch
import torch.nn as nn

from utils.positional_encoding import SinusoidalPositionEmbeddings, apply_pos_encoding


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]  # type: ignore

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)  # type: ignore
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class AttentionStftDecoder(nn.Module):
    """
    Inspired by WavTokenizer, this is a decoder module that uses inverse STFT to map VQ tokens into waveforms.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        num_layers: int,
        num_heads: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        dropout: float = 0.1,
        padding: Literal["center", "same"] = "same",
    ) -> None:
        """
        Initializes the AttentionSTFT class.

        Args:
            hidden_dim (int): The dimension of the hidden layers. Must be an even number.
            input_dim (int): The dimension of the input features.
            num_layers (int): The number of layers in the transformer encoder.
            num_heads (int): The number of attention heads in the transformer encoder.
            n_fft (int): The number of FFT components.
            hop_length (int): The hop length for the ISTFT.
            win_length (int): The window length for the ISTFT.
            dropout (float, optional): The dropout rate. Default is 0.1.
            padding (Literal["center", "same"], optional): The padding type for the ISTFT. Default is "same".

        Raises:
            ValueError: If `hidden_dim` is not an even number.
            ValueError: If `hop_length` is greater than `win_length`.
        """
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden dimension must be even.")
        if hop_length > win_length:
            raise ValueError("Hop length must be less than or equal to window length.")
        self._hidden_dim = hidden_dim
        self._input_dim = input_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout

        self._before_istft_dim = n_fft // 2 + 1
        self.istft = ISTFT(n_fft, hop_length, win_length, padding)
        self._in_projection = nn.Linear(input_dim, hidden_dim)
        self._before_istft_projection = nn.Linear(
            hidden_dim, self._before_istft_dim * 2
        )
        self._pos_encoding = SinusoidalPositionEmbeddings(hidden_dim)

        # Transformers
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            norm_first=True,
            batch_first=True,
            dropout=dropout,
        )
        self._transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Output tensor after applying the inverse short-time Fourier transform (iSTFT),
                            with shape (batch_size, 1, length).
        """
        z = self._in_projection(
            z.transpose(1, 2).contiguous()
        )  # z: BS x C x Len -> BS x Len x H
        z = apply_pos_encoding(z, self._pos_encoding)
        z = self._transformer_encoder(z)
        z = self._before_istft_projection(z)
        z = (
            z[..., : self._before_istft_dim] + 1j * z[..., self._before_istft_dim :]
        )  # Needs to be complex
        z = self.istft(z.transpose(1, 2).contiguous())
        return z.unsqueeze(1)
