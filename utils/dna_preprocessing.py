import numpy as np
import pandas as pd
import torch
from typing import List, Union

# DNA to one-hot encoding map
DNA_MAP = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "C": [0, 0, 0, 1],
    "N": [0, 0, 0, 0]  # unknown/ambiguous base
}


def one_hot_encode_sequence(seq: str, seq_len: int) -> np.ndarray:
    """
    Convert one DNA sequence into one-hot encoded matrix of shape (seq_len, 4).
    Sequences longer than seq_len are truncated, shorter ones are padded.
    """
    seq = seq.upper()
    arr = [DNA_MAP.get(base, [0, 0, 0, 0]) for base in seq]
    
    # pad or truncate
    if len(arr) < seq_len:
        arr += [[0, 0, 0, 0]] * (seq_len - len(arr))
    else:
        arr = arr[:seq_len]
    
    return np.array(arr)


def load_sequences(file_path: str) -> List[str]:
    """
    Load sequences from different file formats:
    - .txt: assumes one sequence per line
    - .csv: assumes one column with sequences (first column if multiple)
    - .fasta / .fa: parses FASTA formatted files
    """
    sequences = []

    if file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            sequences = [line.strip() for line in f if line.strip()]

    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        sequences = df.iloc[:, 0].astype(str).tolist()  # take first column

    elif file_path.endswith((".fasta", ".fa", ".fsa", ".fsa_aa")):
        with open(file_path, "r") as f:
            seq = ""
            for line in f:
                if line.startswith(">"):
                    if seq:
                        sequences.append(seq)
                        seq = ""
                else:
                    seq += line.strip()
            if seq:
                sequences.append(seq)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    return sequences


def load_dataset(file_path: str, seq_len: int = 100) -> torch.Tensor:
    """
    Load DNA dataset from file, convert to one-hot encoding, and return tensor.
    Shape: (num_samples, seq_len, 4)
    """
    sequences = load_sequences(file_path)
    dataset = [one_hot_encode_sequence(seq, seq_len) for seq in sequences]
    dataset = np.array(dataset)

    return torch.tensor(dataset, dtype=torch.float32)
