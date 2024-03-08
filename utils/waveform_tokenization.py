"""
Code from Andrej Karpathy's video about tokenization of LLMs, I'm trying to apply it to audio data.
"""

import torch
import tqdm


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Calculate the frequency of occurrence of consecutive pairs of integers in the given list.

    Args:
    *   ids (list[int]): The list of integers.

    Returns:
    *   dict[tuple[int, int], int]: A dictionary where the keys are tuples of consecutive pairs of integers
        and the values are the frequency of occurrence of each pair.
    """
    counts = {}
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    Replaces all consecutive occurrences of a pair of integers in the given list with a new token.

    Args:
        ids (list[int]): The list of integers.
        pair (tuple[int, int]): The pair of integers to be replaced.
        idx (int): The new token to replace the pair with.

    Returns:
        list[int]: The modified list with the pair replaced by the new token.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def quantize_waveform_256(waveform: torch.Tensor) -> torch.Tensor:
    """
    Quantizes the waveform to 8 bits, saves as a uint8 tensor.

    Args:
        waveform (torch.Tensor): The input waveform.

    Returns:
        torch.Tensor: The quantized waveform.
    """
    return ((waveform + 1) * 127.5).ceil().to(torch.uint8)


def dequantize_waveform_256(waveform: torch.Tensor) -> torch.Tensor:
    """
    Takes a quantized to 8-bit waveform and returns a de-quantized waveform.

    Args:
        waveform (torch.Tensor): The input waveform tensor.

    Returns:
        torch.Tensor: The de-quantized waveform tensor.
    """
    return (waveform.to(torch.float32) / 127.5) - 1


def generate_vocabulary_from_merges_256(
    merges: dict[tuple[int, int], int]
) -> dict[int, list[int]]:
    """
    Generate a vocabulary from merges.

    Args:
        merges (dict[tuple[int, int], int]): A dictionary containing merges.

    Returns:
        dict[int, list[int]]: The generated vocabulary.
    """
    vocab = {idx: list(bytes([idx])) for idx in range(256)}
    vocab = {idx: list(pair) for pair, idx in merges.items()}
    return vocab


def decode_waveform_256(ids: list[int], vocab: dict[int, list[int]]) -> torch.Tensor:
    """
    Decodes a list of integer IDs into a waveform tensor.

    Args:
        ids (list[int]): The list of integer IDs representing the waveform.
        vocab (dict[int, list[int]]): The vocabulary mapping integer IDs to waveform bytes.

    Returns:
        torch.Tensor: The decoded waveform tensor.
    """
    waveform_bytes: list[int] = []
    for idx in ids:
        waveform_bytes.extend(vocab[idx])
    waveform_tensor = torch.tensor(waveform_bytes, dtype=torch.uint8)
    return dequantize_waveform_256(waveform_tensor)


def encode_waveform_256(
    waveform: torch.Tensor, merges: dict[tuple[int, int], int]
) -> list[int]:
    """
    Encodes a waveform into a list of integers using 256 quantization levels.

    Args:
        waveform (torch.Tensor): The input waveform to be encoded.
        merges (dict[tuple[int, int], int]): A dictionary containing merge operations.

    Returns:
        list[int]: The encoded waveform as a list of integers.
    """
    q_waveform = quantize_waveform_256(waveform).tolist()
    while len(q_waveform) >= 2:
        stats = get_stats(q_waveform)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        q_waveform = merge(q_waveform, pair, idx)
    return q_waveform


Merges = dict[tuple[int, int], int]
IDs = list[int]


def merge_and_tokenize_256(
    ids: list[int], num_merges: int, print_progress: bool = False
) -> tuple[IDs, Merges]:
    """
    Merge and tokenize a list of IDs using a specified number of merges.

    Args:
        ids (list[int]): The list of IDs to be merged and tokenized.
        num_merges (int): The number of merges to perform.
        print_progress (bool, optional): Whether to print the progress of the merges. Defaults to False.

    Returns:
        tuple[IDs, Merges]: A tuple containing the merged IDs and the merge dictionary.

    """
    merges = {}  # (int, int) -> int
    for i in tqdm.tqdm(range(num_merges), desc="Generating merges..."):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)  # type: ignore
        idx = 256 + i
        if print_progress:
            print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return ids, merges
