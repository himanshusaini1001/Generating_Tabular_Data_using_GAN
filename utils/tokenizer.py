# utils/tokenizer.py

import torch

class CharTokenizer:
    """
    A simple character-level tokenizer for DNA sequences.
    """
    def __init__(self, sequences):
        # Create a sorted list of unique characters in sequences
        chars = sorted(set("".join(sequences)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, seq, seq_len):
        """
        Convert sequence to list of integers (indices).
        Pads with 0 if sequence is shorter than seq_len.
        """
        arr = [self.char2idx.get(ch, 0) for ch in seq]
        if len(arr) < seq_len:
            arr += [0] * (seq_len - len(arr))  # padding
        return arr[:seq_len]

    def decode(self, arr):
        """
        Convert list of integers back to string sequence.
        """
        if isinstance(arr, torch.Tensor):
            arr = arr.tolist()
        if isinstance(arr, int):
            arr = [arr]
        return "".join([self.idx2char.get(i, "?") for i in arr])
