from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from .utils import GridSpec, check_courant, gaussian_pulse


@dataclass
class SimulationConfig:
    grid: GridSpec
    c: float
    rho: float
    alpha_A: np.ndarray  # damping term
    alpha_B: np.ndarray
    k_coeff: np.ndarray  # boundary coeff
    b_coeff: np.ndarray  # boundary coeff
    beta: float          # admittance
    device: str = "cpu"


class FDTDSimulator:
    def __init__(self, cfg: SimulationConfig, air_mask: np.ndarray | None = None):
        self.cfg = cfg
        check_courant(cfg.grid.dt * cfg.c / cfg.grid.dx)

        # Allocate fields on MPS/CPU
        shape = cfg.grid.shape
        device = torch.device(cfg.device)
        self.p_nm1 = torch.zeros(shape, dtype=torch.float32, device=device)
        self.p_n = torch.zeros(shape, dtype=torch.float32, device=device)
        self.p_np1 = torch.zeros(shape, dtype=torch.float32, device=device)

        # coefficients
        self.alpha_A = torch.from_numpy(cfg.alpha_A).to(device)
        self.alpha_B = torch.from_numpy(cfg.alpha_B).to(device)
        self.k_coeff = torch.from_numpy(cfg.k_coeff).to(device)
        self.b_coeff = torch.from_numpy(cfg.b_coeff).to(device)

        self.device = device
        self.dx = cfg.grid.dx
        self.dt = cfg.grid.dt
        self.c = cfg.c
        self.beta = cfg.beta

        self.lambda_c = cfg.c * cfg.grid.dt / cfg.grid.dx

        if air_mask is None:
            air_mask = np.ones(cfg.grid.shape, dtype=np.float32)
        self.air_mask = torch.from_numpy(air_mask.astype(np.float32)).to(device)
        # Precompute neighbor air masks for mirror boundary at solid cells
        self.mask_xp = torch.empty_like(self.air_mask)
        self.mask_xm = torch.empty_like(self.air_mask)
        self.mask_yp = torch.empty_like(self.air_mask)
        self.mask_ym = torch.empty_like(self.air_mask)
        self.mask_zp = torch.empty_like(self.air_mask)
        self.mask_zm = torch.empty_like(self.air_mask)
        self.mask_xp[:-1, :, :] = self.air_mask[1:, :, :]
        self.mask_xp[-1, :, :] = 0.0
        self.mask_xm[1:, :, :] = self.air_mask[:-1, :, :]
        self.mask_xm[0, :, :] = 0.0
        self.mask_yp[:, :-1, :] = self.air_mask[:, 1:, :]
        self.mask_yp[:, -1, :] = 0.0
        self.mask_ym[:, 1:, :] = self.air_mask[:, :-1, :]
        self.mask_ym[:, 0, :] = 0.0
        self.mask_zp[:, :, :-1] = self.air_mask[:, :, 1:]
        self.mask_zp[:, :, -1] = 0.0
        self.mask_zm[:, :, 1:] = self.air_mask[:, :, :-1]
        self.mask_zm[:, :, 0] = 0.0

    def step(self, source_idx: Tuple[int, int, int], source_val: float):
        # Compute laplacian (6-point stencil)
        p = self.p_n
        # Mirror boundary at internal solids: if neighbor is solid, use current cell (zero gradient)
        xp = torch.empty_like(p)
        xm = torch.empty_like(p)
        yp = torch.empty_like(p)
        ym = torch.empty_like(p)
        zp = torch.empty_like(p)
        zm = torch.empty_like(p)

        xp[:-1, :, :] = p[1:, :, :]
        xp[-1, :, :] = p[-1, :, :]
        xm[1:, :, :] = p[:-1, :, :]
        xm[0, :, :] = p[0, :, :]
        yp[:, :-1, :] = p[:, 1:, :]
        yp[:, -1, :] = p[:, -1, :]
        ym[:, 1:, :] = p[:, :-1, :]
        ym[:, 0, :] = p[:, 0, :]
        zp[:, :, :-1] = p[:, :, 1:]
        zp[:, :, -1] = p[:, :, -1]
        zm[:, :, 1:] = p[:, :, :-1]
        zm[:, :, 0] = p[:, :, 0]

        xp = xp * self.mask_xp + p * (1.0 - self.mask_xp)
        xm = xm * self.mask_xm + p * (1.0 - self.mask_xm)
        yp = yp * self.mask_yp + p * (1.0 - self.mask_yp)
        ym = ym * self.mask_ym + p * (1.0 - self.mask_ym)
        zp = zp * self.mask_zp + p * (1.0 - self.mask_zp)
        zm = zm * self.mask_zm + p * (1.0 - self.mask_zm)
        lap = (xp + xm + yp + ym + zp + zm - 6.0 * p)

        # Eq. (8)-like update (lossy + boundary); simplified for uniform beta & coefficients
        # p_np1 = (1 + b + alpha_A/(2T) + alpha_B*T^2)^{-1} * (
        #   lambda^2 * lap + (2 - k*lambda^2) p + (b - 1 - alpha_A/2) p_nm1 )
        # Note: alpha_A, alpha_B, k, b are spatially varying from PML/BC

        T = self.dt
        lam2 = self.lambda_c ** 2
        denom = (1.0 + self.b_coeff + self.alpha_A / (2.0 * T) + self.alpha_B * (T**2))
        term = (
            lam2 * lap + (2.0 - self.k_coeff * lam2) * p + (self.b_coeff - 1.0 - self.alpha_A / 2.0) * self.p_nm1
        )
        self.p_np1 = term / denom

        # Inject source
        self.p_np1[source_idx] += source_val
        # Enforce solids as rigid blocks
        self.p_np1 = self.p_np1 * self.air_mask

        # Rotate buffers
        self.p_nm1, self.p_n, self.p_np1 = self.p_n, self.p_np1, self.p_nm1

    def run(self, source_idx: Tuple[int, int, int], steps: int, source_fmax: float, record_idx: np.ndarray):
        # record_idx shape (M,3)
        rec = torch.zeros((steps, record_idx.shape[0]), device=self.device)
        for n in range(steps):
            t = n * self.dt
            s = gaussian_pulse(t, source_fmax)
            self.step(source_idx, s)
            # sample
            vals = self.p_n[record_idx[:,0], record_idx[:,1], record_idx[:,2]]
            rec[n, :] = vals
        return rec.detach().cpu().numpy()


def build_simulation(grid: GridSpec, alpha_A, alpha_B, k_coeff, b_coeff, c, rho, beta, device="cpu", air_mask=None):
    cfg = SimulationConfig(
        grid=grid,
        c=c,
        rho=rho,
        alpha_A=alpha_A.astype(np.float32),
        alpha_B=alpha_B.astype(np.float32),
        k_coeff=k_coeff.astype(np.float32),
        b_coeff=b_coeff.astype(np.float32),
        beta=beta,
        device=device,
    )
    return FDTDSimulator(cfg, air_mask=air_mask)
