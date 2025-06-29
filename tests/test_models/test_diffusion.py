import torch

from models.models.diffusion.token import TokenDiffusionTransformer
from models.modules.diffusion_llm import DiffusionLLMLightningModule


def test_token_diffusion_transformer_forward_inputs_only(
    token_diffusion_transformer: TokenDiffusionTransformer,
) -> None:
    dummy_input = torch.randint(0, 100, (3, 128, 4))  # BS x L
    dummy_output = token_diffusion_transformer.forward(dummy_input)
    assert dummy_output.shape == (3, 128, 4, 100)  # BS x C x L


def test_token_diffusion_transformer_forward_inputs_only_cuda(
    token_diffusion_transformer: TokenDiffusionTransformer,
) -> None:
    token_diffusion_transformer = token_diffusion_transformer.to("cuda")
    dummy_input = torch.randint(0, 100, (3, 128, 4)).to("cuda")  # BS x L
    dummy_output = token_diffusion_transformer.forward(dummy_input)
    assert dummy_output.shape == (3, 128, 4, 100)  # BS x C x L


def test_token_diffusion_transformer_forward_with_mask(
    token_diffusion_transformer: TokenDiffusionTransformer,
) -> None:
    dummy_input = torch.randint(0, 100, (3, 128, 4))  # BS x L
    dummy_mask = torch.randint(0, 2, (3, 128)).bool()  # BS x L
    dummy_output = token_diffusion_transformer.forward(dummy_input, mask=dummy_mask)
    assert dummy_output.shape == (3, 128, 4, 100)  # BS x C x L


def test_token_diffusion_transformer_forward_with_mask_cuda(
    token_diffusion_transformer: TokenDiffusionTransformer,
) -> None:
    token_diffusion_transformer = token_diffusion_transformer.to("cuda")
    dummy_input = torch.randint(0, 100, (3, 128, 4)).to("cuda")  # BS x L
    dummy_mask = torch.randint(0, 2, (3, 128)).bool().to("cuda")  # BS x L
    dummy_output = token_diffusion_transformer.forward(dummy_input, mask=dummy_mask)
    assert dummy_output.shape == (3, 128, 4, 100)  # BS x C x L


def test_diffusion_module_generate(
    token_diffusion_module: DiffusionLLMLightningModule,
) -> None:
    dummy_input = torch.randint(0, 100, (3, 512, 4))  # BS x L x RS
    dummy_output = token_diffusion_module.generate(init_latent=dummy_input)
    assert dummy_output.shape == (3, 512, 4)  # BS x L x RS


def test_diffusion_module_generate_cuda(
    token_diffusion_module: DiffusionLLMLightningModule,
) -> None:
    token_diffusion_module = token_diffusion_module.to("cuda")
    dummy_input = torch.randint(0, 100, (3, 512, 4)).to("cuda")  # BS x L x RS
    dummy_output = token_diffusion_module.generate(init_latent=dummy_input)
    assert dummy_output.shape == (3, 512, 4)  # BS x L x RS
