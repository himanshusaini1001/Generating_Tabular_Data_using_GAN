import torch
import torch.nn as nn

class Discriminator1(nn.Module):
    def __init__(self, vocab_size, hidden_dim, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * seq_len, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        return torch.sigmoid(self.fc(out))


class Discriminator2(nn.Module):
    def __init__(self, vocab_size, hidden_dim, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * seq_len, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        return torch.sigmoid(self.fc(out))
