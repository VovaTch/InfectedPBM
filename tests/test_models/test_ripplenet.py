import torch
import pytest
from models.multi_level_vqvae.ripplenet import RippleReconstructor, ripple_linear_func


@pytest.fixture
def ripple_linear_input() -> torch.Tensor:
    return torch.randn(10, 20)  # Example input tensor


@pytest.fixture
def ripple_linear_weight() -> torch.Tensor:
    return torch.randn(30, 20, 2)  # Example weight tensor


@pytest.fixture
def ripple_linear_bias() -> torch.Tensor:
    return torch.randn(30, 21)  # Example bias tensor


def test_ripple_linear(
    ripple_linear_input: torch.Tensor,
    ripple_linear_weight: torch.Tensor,
    ripple_linear_bias: torch.Tensor,
) -> None:
    out_features = ripple_linear_bias.size(0)
    expected_output_size = (ripple_linear_input.size(0), out_features)

    output = ripple_linear_func(
        ripple_linear_input, out_features, ripple_linear_weight, ripple_linear_bias
    )

    assert output.size() == expected_output_size


@pytest.fixture
def ripple_reconstructor_input() -> torch.Tensor:
    return torch.randn(10, 1)  # Example input tensor


@pytest.fixture
def ripple_reconstructor_model() -> RippleReconstructor:
    return RippleReconstructor(hidden_size=20, num_inner_layers=2)


def test_ripple_reconstructor_forward(
    ripple_reconstructor_input: torch.Tensor,
    ripple_reconstructor_model: RippleReconstructor,
) -> None:
    output = ripple_reconstructor_model(ripple_reconstructor_input)
    assert output.shape == (10, 1)


def test_ripple_reconstructor_forward_gpu(
    ripple_reconstructor_input: torch.Tensor,
    ripple_reconstructor_model: RippleReconstructor,
) -> None:
    ripple_reconstructor_model = ripple_reconstructor_model.to("cuda")
    ripple_reconstructor_input = ripple_reconstructor_input.to("cuda")
    output = ripple_reconstructor_model(ripple_reconstructor_input)
    assert output.shape == (10, 1)
