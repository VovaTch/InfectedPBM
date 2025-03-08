from abc import ABC, abstractmethod

import torch.nn as nn


class Decoder(ABC, nn.Module):
    """
    Base class for a decoder model. This is a pytorch tensor module that also can return the last layer.
    """

    @property
    @abstractmethod
    def last_layer(self) -> nn.Module:
        """
        Returns the last layer of the decoder.

        Returns:
            nn.Module: The last layer of the decoder.
        """
        ...
