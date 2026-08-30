# model.py — TCN for sequence-to-sequence regression (Python 3.8+)
from typing import List
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.chomp == 0 else x[:, :, :-self.chomp].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, d: int, drop: float):
        super().__init__()
        pad = (k - 1) * d
        self.c1 = weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d))
        self.h1 = Chomp1d(pad); self.a1 = nn.ReLU(); self.d1 = nn.Dropout(drop)
        self.c2 = weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d))
        self.h2 = Chomp1d(pad); self.a2 = nn.ReLU(); self.d2 = nn.Dropout(drop)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.a_out = nn.ReLU()
        # init
        nn.init.kaiming_normal_(self.c1.weight, nonlinearity='relu'); nn.init.zeros_(self.c1.bias)
        nn.init.kaiming_normal_(self.c2.weight, nonlinearity='relu'); nn.init.zeros_(self.c2.bias)
        if self.down is not None:
            nn.init.kaiming_normal_(self.down.weight, nonlinearity='linear'); nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.d1(self.a1(self.h1(self.c1(x))))
        y = self.d2(self.a2(self.h2(self.c2(y))))
        res = x if self.down is None else self.down(x)
        return self.a_out(y + res)

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels: int, channels: List[int], k: int = 3, drop: float = 0.1):
        super().__init__()
        layers = []
        c = in_channels
        for i, co in enumerate(channels):
            layers.append(TemporalBlock(c, co, k, d=2**i, drop=drop))
            c = co
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TCNRegressor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 channels: List[int] = (64, 64, 128, 128),
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tcn = TemporalConvNet(in_channels, list(channels), kernel_size, dropout)
        self.head = nn.Conv1d(channels[-1], out_channels, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tcn(x))
