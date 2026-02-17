import torch
import torch.nn as nn

class Generator1(nn.Module):
    def __init__(self, z_dim, hidden_dim, seq_len, vocab_size):
        super().__init__()
        self.fc = nn.Linear(z_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, z):
        batch = z.size(0)
        x = torch.relu(self.fc(z)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return torch.softmax(self.out(out), dim=-1)


class Generator2(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq_probs):
        out, _ = self.lstm(seq_probs)
        return torch.softmax(self.out(out), dim=-1)
