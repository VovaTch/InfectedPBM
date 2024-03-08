import torch
from utils.waveform_tokenization import (
    decode_waveform_256,
    dequantize_waveform_256,
    encode_waveform_256,
    generate_merges_256,
)


def test_generate_merges_256() -> None:
    ids = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 2, 3]
    num_merges = 2
    merges = generate_merges_256(ids, num_merges)
    assert merges == {(0, 0): 256, (2, 3): 257}


def test_encode_waveform_256() -> None:
    ids = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 2, 3]
    waveform = dequantize_waveform_256(torch.tensor(ids, dtype=torch.uint8))
    merges = {(0, 0): 256, (2, 3): 257}
    tokens = encode_waveform_256(waveform, merges)  # type: ignore
    assert tokens == [256, 1, 1, 257, 4, 5, 6, 7, 8, 9, 256, 257]


def test_decode_waveform_256() -> None:
    ids = [256, 1, 1, 257, 4, 5, 6, 7, 8, 9, 256, 257]
    ids_extended = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 2, 3]
    vocab = {idx: [idx] for idx in range(256)}
    vocab.update({256: [0, 0], 257: [2, 3]})
    decoded_waveform = decode_waveform_256(ids, vocab)
    waveform = dequantize_waveform_256(torch.tensor(ids_extended, dtype=torch.uint8))
    assert torch.allclose(waveform, decoded_waveform)
