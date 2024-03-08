import torch


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    counts = {}
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
    new_ids = []
    i = 0
    while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def quantize_waveform_256(waveform: torch.Tensor) -> torch.Tensor:
    return ((waveform + 1) * 127.5).ceil().to(torch.uint8)


def dequantize_waveform_256(waveform: torch.Tensor) -> torch.Tensor:
    return (waveform.to(torch.float32) / 127.5) - 1


def generate_vocabulary_from_merges_256(
    merges: dict[tuple[int, int], int]
) -> dict[int, list[int]]:
    vocab = {idx: list(bytes([idx])) for idx in range(256)}
    vocab = {idx: list(pair) for pair, idx in merges.items()}
    return vocab


def decode_waveform_256(ids: list[int], vocab: dict[int, list[int]]) -> torch.Tensor:
    # given ids (list of integers), return a waveform (PyTorch tensor)
    waveform_bytes: list[int] = []
    for idx in ids:
        waveform_bytes.extend(vocab[idx])
    waveform_tensor = torch.tensor(waveform_bytes, dtype=torch.uint8)
    return dequantize_waveform_256(waveform_tensor)


def encode_waveform_256(
    waveform: torch.Tensor, merges: dict[tuple[int, int], int]
) -> list[int]:
    # given a waveform (PyTorch tensor), return list of integers (the tokens)
    q_waveform = quantize_waveform_256(waveform).tolist()
    while len(q_waveform) >= 2:
        stats = get_stats(q_waveform)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        q_waveform = merge(q_waveform, pair, idx)
    return q_waveform


def generate_merges_256(ids: list[int], num_merges: int) -> dict[tuple[int, int], int]:
    """
    Generates a dictionary of merges for waveform tokenization.

    Args:
        ids (list[int]): The list of token IDs.
        num_merges (int): The number of merges to perform.

    Returns:
        dict[tuple[int, int], int]: A dictionary mapping pairs of token IDs to their merged token ID.
    """
    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)  # type: ignore
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges
