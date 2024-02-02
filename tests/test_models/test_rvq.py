import torch
import pytest
from models.vq_codebook import ResidualCodebookCollection


@pytest.fixture
def codebook_collection():
    token_dim = 64
    num_codebooks = 4
    num_tokens = 256
    return ResidualCodebookCollection(token_dim, num_codebooks, num_tokens)


def test_apply_codebook(codebook_collection: ResidualCodebookCollection) -> None:
    x_in = torch.randn(10, 64, 2)
    z_q, indices = codebook_collection.apply_codebook(x_in)
    assert z_q.shape == x_in.shape
    assert indices.shape == (
        10,
        2,
        4,
    )


def test_embed_codebook(codebook_collection: ResidualCodebookCollection) -> None:
    indices = torch.randint(64, size=(10, 2, 4))
    emb = codebook_collection.embed_codebook(indices)
    assert emb.shape == (10, 64, 2)


def test_update_usage(codebook_collection: ResidualCodebookCollection) -> None:
    min_enc = torch.randint(64, size=(10, 2, 4))
    codebook_collection.update_usage(min_enc)
    # Add assertions here to check if the usage is updated correctly


def test_reset_usage(codebook_collection: ResidualCodebookCollection) -> None:
    codebook_collection.reset_usage()
    # Add assertions here to check if the usage is reset correctly


def test_random_restart(codebook_collection: ResidualCodebookCollection) -> None:
    dead_codes = codebook_collection.random_restart()
    # Add assertions here to check if the dead codes are counted correctly
