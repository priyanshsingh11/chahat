"""
SF-IDS MVP - Model: hybrid DCNN-BiLSTM (after Hnamte & Hussain, 2023)
CNN layers extract local patterns across the flow features, BiLSTM models the
resulting sequence forwards and backwards, dense + softmax classifies.
"""
import torch
import torch.nn as nn


class DCNNBiLSTM(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Deep CNN block: [B, 1, F] -> [B, 64, F/4]
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # BiLSTM over the CNN feature sequence
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64,
                              batch_first=True, bidirectional=True)
        # Dense classifier (softmax is applied inside CrossEntropyLoss / at inference)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):                 # x: [B, F]
        x = x.unsqueeze(1)                # [B, 1, F]
        x = self.cnn(x)                   # [B, 64, L]
        x = x.permute(0, 2, 1)            # [B, L, 64] for the LSTM
        _, (h, _) = self.bilstm(x)        # h: [2, B, 64]
        x = torch.cat([h[0], h[1]], dim=1)  # [B, 128] forward+backward final states
        return self.head(x)


if __name__ == "__main__":
    m = DCNNBiLSTM(num_features=78, num_classes=4)
    out = m(torch.randn(8, 78))
    print("Forward pass OK:", out.shape)
    print("Parameters:", sum(p.numel() for p in m.parameters()))
