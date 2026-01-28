from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.interpolate import RectBivariateSpline


class ConvSame(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2)

    def forward(self, x):
        y = self.conv(x)
        if y.shape[-2:] != x.shape[-2:]:
            H, W = x.shape[-2:]
            y = y[..., :H, :W]
        return y


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, k: int):
        super().__init__()
        self.conv1 = ConvSame(channels, channels, k)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.95)
        self.prelu = nn.PReLU()
        self.conv2 = ConvSame(channels, channels, k)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.95)

    def forward(self, x):
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.prelu(out + x)


class DPGNet(nn.Module):
    def __init__(self, in_ch=4, out_actions=3):
        super().__init__()
        # Based on Fig.7: 4x4 residual blocks with 16 channels, then conv layers to 3 channels
        self.stem = ConvSame(in_ch, 16, 3)
        self.rb1 = ResidualBlock(16, 4)
        self.rb2 = ResidualBlock(16, 4)
        self.rb3 = ResidualBlock(16, 4)
        self.rb4 = ResidualBlock(16, 4)
        self.conv1 = ConvSame(16, 32, 2)
        self.conv2 = ConvSame(32, 16, 2)
        self.conv3 = ConvSame(16, 3, 2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.prelu(self.stem(x))
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.conv2(x))
        x = self.conv3(x)
        # Softmax over action channel
        return F.softmax(x, dim=1)


def sample_action(probs: torch.Tensor):
    # probs: (B,3,H,W) -> action in {-1,0,1}, log-prob sum, entropy sum
    B, C, H, W = probs.shape
    flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
    dist = torch.distributions.Categorical(probs=flat)
    a = dist.sample()
    logp = dist.log_prob(a)
    entropy = dist.entropy()
    a = a.reshape(B, H, W)
    logp = logp.reshape(B, H, W)
    entropy = entropy.reshape(B, H, W)
    return a - 1, logp.sum(dim=(1, 2)), entropy.sum(dim=(1, 2))


def build_input(pattern: np.ndarray) -> np.ndarray:
    # pattern: (H,W)
    # channels: pattern, 2D FFT, 2D cepstrum, autocorrelation (downsampled)
    H, W = pattern.shape
    p = pattern.astype(np.float32)
    fft2 = np.abs(np.fft.fftshift(np.fft.fft2(p)))
    cep = np.abs(np.fft.ifft2(np.log(np.abs(np.fft.fft2(p)) + 1e-6)))
    # 2D autocorrelation (full -> 2H-1 x 2W-1)
    ac_full = signal.correlate2d(p, p, mode="full")
    # interpolate to (H,W) with spline order 5 (RectBivariateSpline)
    y_full = np.linspace(0, 1, ac_full.shape[0])
    x_full = np.linspace(0, 1, ac_full.shape[1])
    y_tgt = np.linspace(0, 1, H)
    x_tgt = np.linspace(0, 1, W)
    spline = RectBivariateSpline(y_full, x_full, ac_full, kx=5, ky=5)
    ac_resized = spline(y_tgt, x_tgt)
    stacked = np.stack([p, fft2, cep, ac_resized], axis=0).astype(np.float32)
    return stacked
