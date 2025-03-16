import torch
from models.models.diffusion.token import TokenDiffusionTransformer


def test_token_diffusion_transformer_forward_inputs_only(
    token_diffusion_transformer: TokenDiffusionTransformer,
) -> None:
    dummy_input = torch.randint(0, 100, (3, 128, 4))  # BS x L
    dummy_output = token_diffusion_transformer.forward(dummy_input)
    assert dummy_output.shape == (3, 128, 4, 100)  # BS x C x L


def test_token_diffusion_transformer_forward_with_mask(
    token_diffusion_transformer: TokenDiffusionTransformer,
) -> None:
    dummy_input = torch.randint(0, 100, (3, 128, 4))  # BS x L
    dummy_mask = torch.randint(0, 2, (3, 128)).bool()  # BS x L
    dummy_output = token_diffusion_transformer.forward(dummy_input, mask=dummy_mask)
    assert dummy_output.shape == (3, 128, 4, 100)  # BS x C x L
