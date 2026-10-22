"""1D CNN over (beat, beat - patient template), fused with RR + template features."""
import torch
from torch import nn


def conv_block(c_in, c_out, k):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class BeatNet(nn.Module):
    def __init__(self, in_ch=2, n_feat=11, n_classes=4, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(in_ch, 16, 7),   # 256 -> 128
            conv_block(16, 32, 5),      # 128 -> 64
            conv_block(32, 64, 5),      # 64  -> 32
            conv_block(64, 64, 3),      # 32  -> 16
            nn.AdaptiveAvgPool1d(1),
        )
        self.feat = nn.Sequential(nn.Linear(n_feat, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, f):
        z = self.cnn(x).squeeze(-1)
        return self.head(torch.cat([z, self.feat(f)], dim=1))
