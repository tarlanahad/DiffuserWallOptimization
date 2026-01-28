from __future__ import annotations

import numpy as np


def polar_response(impulses: np.ndarray, fs: float, fmax: float) -> np.ndarray:
    """
    Compute frequency response magnitude at each mic, limited to fmax.
    impulses: (T, M)
    returns: (M, F) magnitudes
    """
    T, M = impulses.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    idx = freqs <= fmax
    fft = np.fft.rfft(impulses, axis=0)[idx, :]
    mag = np.abs(fft)
    return mag.T  # (M, F)


def autocorrelation_diffusion_coefficient(polar_mag: np.ndarray) -> float:
    """
    Diffusion coefficient from polar response magnitude, averaged over frequency.
    Formula is the standard diffusion coefficient across angles.
    """
    M, F = polar_mag.shape
    vals = []
    for f in range(F):
        I = polar_mag[:, f] ** 2
        sum_I = np.sum(I)
        sum_I2 = np.sum(I**2)
        denom = (M - 1) * sum_I2 + 1e-12
        d = (sum_I**2 - sum_I2) / denom
        vals.append(d)
    return float(np.mean(vals))
