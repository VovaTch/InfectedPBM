from abc import ABC, abstractmethod

import torch.nn as nn


class DiscriminatorHead(ABC, nn.Module):
    """
    Base class for discriminator head component, to get the last layer from the discriminator
    """

    @property
    @abstractmethod
    def last_layer(self) -> nn.Module:
        """
        Returns the last layer of the network for computing adaptive discriminator weights

        Returns:
            nn.Module: Last layer of the net
        """
        ...


class Discriminator(ABC, nn.Module):
    """
    Base class for a discriminator model, includes the classification head where we get the
    last layer from the discriminator to compute adaptive discriminator weight

    Attributes:
        class_head (DiscriminatorHead): The classification head of the discriminator
    """

    class_head: DiscriminatorHead

    @property
    def last_layer(self) -> nn.Module:
        """
        Returns the last layer of the class head.

        This method accesses the `last_layer` attribute of the `class_head`
        and returns it. The `last_layer` is expected to be an instance of
        `nn.Module`.

        Returns:
            nn.Module: The last layer of the class head.
        """
        return self.class_head.last_layer
