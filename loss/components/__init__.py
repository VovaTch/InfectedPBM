from .classification import BasicClassificationLoss, PercentCorrect
from .codebook import AlignLoss, CommitLoss
from .lm_cross_entropy import LLMClassificationLoss, TokenEntropy, LLMPercentCorrect
from .mel_spec import MelSpecLoss, MelSpecDiffusionLoss
from .reconstruction import RecLoss, NoisePredLoss, DiffReconstructionLoss, EdgeRecLoss
from .base import LossComponent
