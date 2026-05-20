import torch.nn as nn


class BiLSTMDecoder(nn.Module):
    def __init__(
        self,
        gesture_vocab,
        char_vocab,
        embed_dim=128,
        hidden=256,
        num_layers=1,
        dropout=0.0
    ):
        super().__init__()

        self.embedding = nn.Embedding(gesture_vocab, embed_dim)

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 512),
            nn.ReLU(),
            nn.Linear(512, char_vocab)
        )

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x