import numpy as np
import torch

# -------------------------
# Tokenizer for sequences
# -------------------------
class CharTokenizer:
    def __init__(self, sequences):
        chars = sorted(set("".join(sequences)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, seq, seq_len):
        arr = [self.char2idx.get(ch, 0) for ch in seq]
        if len(arr) < seq_len:
            arr = arr + [0] * (seq_len - len(arr))  # padding
        return arr[:seq_len]

    def decode(self, arr):
        # If arr is a single int, convert to list
        if isinstance(arr, int):
            arr = [arr]
        return "".join([self.idx2char.get(i, "?") for i in arr])

# -------------------------
# Prepare dataset
# -------------------------
def prepare_dataset(sequences, seq_len):
    tokenizer = CharTokenizer(sequences)
    encoded = [tokenizer.encode(seq, seq_len) for seq in sequences]
    dataset = torch.tensor(np.array(encoded))
    return dataset, tokenizer
