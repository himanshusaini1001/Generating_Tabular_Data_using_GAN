# utils/sequence_heatmap.py

import numpy as np
import matplotlib.pyplot as plt

class SequenceHeatmap:
    def __init__(self, file_path):
        """
        Initialize with a path to a .txt file containing sequences.
        Each line should be a sequence (e.g., genome data).
        """
        self.file_path = file_path
        self.sequences = self._load_sequences()
        self.nucleotides = ['A', 'C', 'G', 'T']
        self.mapping = {n: i for i, n in enumerate(self.nucleotides)}
        self.seq_len = len(self.sequences[0])
        self.freq_matrix = self._compute_frequency_matrix()

    def _load_sequences(self):
        sequences = []
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip().upper()
                if line:
                    sequences.append(line)
        if not sequences:
            raise ValueError("No sequences found in the file!")
        return sequences

    def _compute_frequency_matrix(self):
        freq_matrix = np.zeros((4, self.seq_len))  # 4 nucleotides x sequence length
        for seq in self.sequences:
            for pos, ch in enumerate(seq):
                if ch in self.mapping:
                    freq_matrix[self.mapping[ch], pos] += 1
        return freq_matrix

    def get_frequency_matrix(self):
        """
        Returns the frequency matrix (shape: 4 x seq_len)
        """
        return self.freq_matrix

    def plot_heatmap(self, title="Nucleotide Frequency Heatmap"):
        """
        Plots a heatmap from the frequency matrix.
        """
        plt.figure(figsize=(12, 4))
        plt.imshow(self.freq_matrix, cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Count')
        plt.yticks(range(4), self.nucleotides)
        plt.xticks(range(self.seq_len))
        plt.xlabel('Position in sequence')
        plt.ylabel('Nucleotide')
        plt.title(title)
        plt.show()
