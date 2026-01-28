from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .utils import GridSpec, speed_of_sound, air_density
from .pml import pml_alpha_profile
from .fdtd import build_simulation
from .geometry import make_diffuser_pattern, apply_action, build_air_mask
from .metrics import polar_response, autocorrelation_diffusion_coefficient
from .rl import DPGNet, build_input, sample_action
from .ga import evolve_population

_warned_fs = False


@dataclass
class ExperimentConfig:
    seed: int
    device: str
    pattern_shape: Tuple[int, int]
    pattern_min: int
    pattern_max: int
    max_depth_cm: float
    cell_size_cm: float
    domain_size_m: Tuple[float, float, float]
    dx_m: float
    courant: float
    fs_hz: float
    sim_time_s: float
    bandwidth_hz: float
    air_temp_c: float
    air_pressure_hpa: float
    air_humidity: float
    pml_thickness: int
    pml_alpha_max: float
    pml_order: float
    rigid_beta: float
    mic_radius_m: float
    mic_count: int
    incidence_angle_deg: float
    source_distance_m: float
    rl: dict
    ga: dict
    log_dir: str


@dataclass
class ExperimentContext:
    cfg: ExperimentConfig
    grid: GridSpec
    alpha_A: np.ndarray
    alpha_B: np.ndarray
    k: np.ndarray
    b: np.ndarray
    c: float
    rho: float
    diffuser_x: int
    center_yz: Tuple[int, int]
    mics_xy: np.ndarray
    mics_xz: np.ndarray
    source: Tuple[int, int, int]
    cache: dict


def load_config(path: str) -> ExperimentConfig:
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return ExperimentConfig(**cfg)


def build_grid(cfg: ExperimentConfig) -> GridSpec:
    nx = int(round(cfg.domain_size_m[0] / cfg.dx_m))
    ny = int(round(cfg.domain_size_m[1] / cfg.dx_m))
    nz = int(round(cfg.domain_size_m[2] / cfg.dx_m))
    c = speed_of_sound(cfg.air_temp_c)
    dt = cfg.courant * cfg.dx_m / c
    fs_effective = 1.0 / dt
    global _warned_fs
    if abs(fs_effective - cfg.fs_hz) / cfg.fs_hz > 0.05 and not _warned_fs:
        print(f"Warning: Effective fs {fs_effective:.1f} != config fs {cfg.fs_hz:.1f}")
        _warned_fs = True
    return GridSpec(nx, ny, nz, cfg.dx_m, dt)


def build_coefficients(cfg: ExperimentConfig, grid: GridSpec):
    # alpha profile
    alpha = pml_alpha_profile(grid.shape, cfg.pml_thickness, cfg.pml_alpha_max, cfg.pml_order)
    # Simplified mapping alpha -> alpha_A, alpha_B (paper assumes alpha = alpha*)
    c = speed_of_sound(cfg.air_temp_c)
    rho = air_density(cfg.air_temp_c, cfg.air_pressure_hpa)
    alpha_A = alpha * (rho * c * c + 1.0 / rho)
    alpha_B = (c * c) * (alpha**2)

    # k coeff: 6 in interior, less near boundaries (vectorized)
    k = np.full(grid.shape, 6.0, dtype=np.float32)
    x_idx = np.arange(grid.nx)[:, None, None]
    y_idx = np.arange(grid.ny)[None, :, None]
    z_idx = np.arange(grid.nz)[None, None, :]
    boundary_count = (
        (x_idx == 0) | (x_idx == grid.nx - 1)
    ).astype(np.int32) + (
        (y_idx == 0) | (y_idx == grid.ny - 1)
    ).astype(np.int32) + (
        (z_idx == 0) | (z_idx == grid.nz - 1)
    ).astype(np.int32)
    k = k - boundary_count
    # b coeff (Eq.8) simplified for uniform beta
    lam = c * grid.dt / grid.dx
    b = (6 - k) * lam * cfg.rigid_beta / 2.0

    return alpha_A, alpha_B, k, b, c, rho


def build_mic_positions(cfg: ExperimentConfig, grid: GridSpec, center: Tuple[int, int, int]):
    # Semicircle in XZ plane (yz plane diffusion), and XY plane (xy diffusion)
    cx, cy, cz = center
    r = cfg.mic_radius_m / cfg.dx_m
    angles = np.linspace(-math.pi/2, math.pi/2, cfg.mic_count)
    coords_xz = []
    coords_xy = []
    for a in angles:
        # In front of diffuser: x <= cx
        x = int(round(cx - r * math.cos(a)))
        z = int(round(cz + r * math.sin(a)))
        coords_xz.append((x, cy, z))
        y = int(round(cy + r * math.sin(a)))
        coords_xy.append((x, y, cz))
    return np.array(coords_xy, dtype=np.int64), np.array(coords_xz, dtype=np.int64)


def build_source_position(cfg: ExperimentConfig, grid: GridSpec, diffuser_x: int, center_yz: Tuple[int, int]):
    cy, cz = center_yz
    dx = int(round(cfg.source_distance_m / cfg.dx_m))
    x = diffuser_x - dx
    if x < cfg.pml_thickness + 2:
        raise ValueError("Source outside domain; adjust domain or diffuser position.")
    return (x, cy, cz)


def choose_diffuser_plane(cfg: ExperimentConfig, grid: GridSpec) -> int:
    # place diffuser so source is inside left side with PML margin
    src_margin = cfg.pml_thickness + 2
    dx = int(round(cfg.source_distance_m / cfg.dx_m))
    x0 = src_margin + dx
    if x0 >= grid.nx - cfg.pml_thickness - 2:
        raise ValueError("Domain too small for source distance and PML.")
    return x0


def build_context(cfg: ExperimentConfig) -> ExperimentContext:
    grid = build_grid(cfg)
    alpha_A, alpha_B, k, b, c, rho = build_coefficients(cfg, grid)
    diffuser_x = choose_diffuser_plane(cfg, grid)
    cy = grid.ny // 2
    cz = grid.nz // 2
    mics_xy, mics_xz = build_mic_positions(cfg, grid, (diffuser_x, cy, cz))
    source = build_source_position(cfg, grid, diffuser_x, (cy, cz))
    return ExperimentContext(
        cfg=cfg,
        grid=grid,
        alpha_A=alpha_A,
        alpha_B=alpha_B,
        k=k,
        b=b,
        c=c,
        rho=rho,
        diffuser_x=diffuser_x,
        center_yz=(cy, cz),
        mics_xy=mics_xy,
        mics_xz=mics_xz,
        source=source,
        cache={},
    )


def evaluate_pattern(ctx: ExperimentContext, pattern: np.ndarray):
    cfg = ctx.cfg
    grid = ctx.grid
    key = pattern.tobytes()
    if key in ctx.cache:
        return ctx.cache[key]
    air_mask = build_air_mask(grid.shape, pattern, cfg.dx_m, cfg.cell_size_cm, ctx.diffuser_x, ctx.center_yz)

    sim = build_simulation(grid, ctx.alpha_A, ctx.alpha_B, ctx.k, ctx.b, ctx.c, ctx.rho, cfg.rigid_beta, device=cfg.device, air_mask=air_mask)
    steps = int(round(cfg.sim_time_s / grid.dt))

    # run simulation and compute diffusion coefficient
    impulses_xy = sim.run(ctx.source, steps, cfg.bandwidth_hz, ctx.mics_xy)
    impulses_xz = sim.run(ctx.source, steps, cfg.bandwidth_hz, ctx.mics_xz)
    fs = 1.0 / grid.dt
    polar_xy = polar_response(impulses_xy, fs, cfg.bandwidth_hz)
    polar_xz = polar_response(impulses_xz, fs, cfg.bandwidth_hz)
    diff_xy = autocorrelation_diffusion_coefficient(polar_xy)
    diff_xz = autocorrelation_diffusion_coefficient(polar_xz)
    avg = 0.5 * (diff_xy + diff_xz)
    ctx.cache[key] = (diff_xy, diff_xz, avg)
    return diff_xy, diff_xz, avg


def run_ga(cfg: ExperimentConfig):
    ctx = build_context(cfg)
    rng = np.random.default_rng(cfg.seed)
    pop = [make_diffuser_pattern(rng, cfg.pattern_shape, cfg.pattern_min, cfg.pattern_max) for _ in range(cfg.ga['population'])]
    log_each_gen = bool(cfg.ga.get("log_each_gen", True))
    for gen in range(cfg.ga['generations']):
        fitness = [evaluate_pattern(ctx, p)[2] for p in pop]
        if log_each_gen:
            best = float(np.max(fitness))
            avg = float(np.mean(fitness))
            print(f"[GA] gen {gen+1}/{cfg.ga['generations']} best={best:.6f} avg={avg:.6f}", flush=True)
        pop = evolve_population(pop, fitness, rng, cfg.ga['swap_prob'], cfg.ga['mutate_prob'], cfg.pattern_min, cfg.pattern_max)
    # return best
    fitness = [evaluate_pattern(ctx, p)[2] for p in pop]
    best_idx = int(np.argmax(fitness))
    return pop[best_idx], fitness[best_idx]


def run_best(cfg: ExperimentConfig):
    # GA seed + DPG best-pool refinement
    ctx = build_context(cfg)
    best_ga, best_score = run_ga(cfg)
    # seed pool with GA best + randoms
    rng = np.random.default_rng(cfg.seed)
    pool = [best_ga] + [make_diffuser_pattern(rng, cfg.pattern_shape, cfg.pattern_min, cfg.pattern_max) for _ in range(cfg.rl['best_pool'] - 1)]

    # Use DPG with best_pool seeded
    device = torch.device(cfg.device)
    net = DPGNet(in_ch=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.rl['lr'])
    episodes = int(cfg.rl.get("episodes", 10))
    steps_per_episode = int(cfg.rl.get("steps_per_episode", 10))
    epochs_per_step = int(cfg.rl.get("epochs_per_step", 1))
    baseline_decay = float(cfg.rl.get("baseline_decay", 0.9))
    entropy_coef = float(cfg.rl.get("entropy_coef", 0.01))
    log_each_step = bool(cfg.rl.get("log_each_step", True))

    baseline = 0.0
    for ep in range(episodes):
        pattern = pool[rng.integers(0, len(pool))]
        prev_xy, prev_xz, prev_score = evaluate_pattern(ctx, pattern)
        for step in range(steps_per_episode):
            inp = build_input(pattern)
            inp_t = torch.from_numpy(inp[None, ...]).to(device)
            probs = net(inp_t)
            action, logp, entropy = sample_action(probs)
            action = action.squeeze(0).cpu().numpy()
            pattern_new = apply_action(pattern, action, cfg.pattern_min, cfg.pattern_max)
            dxy, dxz, score = evaluate_pattern(ctx, pattern_new)
            reward = 0.5 * ((dxy - prev_xy) + (dxz - prev_xz))
            baseline = baseline_decay * baseline + (1.0 - baseline_decay) * reward
            advantage = reward - baseline

            for _ in range(epochs_per_step):
                loss = -(logp.mean() * advantage) - entropy_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            # update pool
            pool.append(pattern_new)
            pool = sorted(pool, key=lambda p: evaluate_pattern(ctx, p)[2], reverse=True)[: cfg.rl['best_pool']]
            pattern = pattern_new
            prev_score = score
            prev_xy = dxy
            prev_xz = dxz
            if log_each_step:
                print(
                    f"[BEST] ep {ep+1}/{episodes} step {step+1}/{steps_per_episode} "
                    f"reward={reward:.6f} dxy={dxy:.6f} dxz={dxz:.6f} avg={score:.6f}",
                    flush=True,
                )

    best = max(pool, key=lambda p: evaluate_pattern(ctx, p)[2])
    return best, evaluate_pattern(ctx, best)[2]


def run_dpg(cfg: ExperimentConfig, use_best_pool: bool = False):
    ctx = build_context(cfg)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    net = DPGNet(in_ch=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.rl['lr'])

    # initialize pool
    pool = [make_diffuser_pattern(rng, cfg.pattern_shape, cfg.pattern_min, cfg.pattern_max) for _ in range(cfg.rl['best_pool'])]

    def pick_start():
        if use_best_pool:
            return pool[rng.integers(0, len(pool))]
        return make_diffuser_pattern(rng, cfg.pattern_shape, cfg.pattern_min, cfg.pattern_max)

    episodes = int(cfg.rl.get("episodes", 10))
    steps_per_episode = int(cfg.rl.get("steps_per_episode", 10))
    epochs_per_step = int(cfg.rl.get("epochs_per_step", 1))
    baseline_decay = float(cfg.rl.get("baseline_decay", 0.9))
    entropy_coef = float(cfg.rl.get("entropy_coef", 0.01))
    log_each_step = bool(cfg.rl.get("log_each_step", True))

    pattern = pick_start()
    prev_xy, prev_xz, prev_score = evaluate_pattern(ctx, pattern)

    baseline = 0.0
    for ep in range(episodes):
        pattern = pick_start()
        prev_xy, prev_xz, prev_score = evaluate_pattern(ctx, pattern)
        for step in range(steps_per_episode):
            inp = build_input(pattern)
            inp_t = torch.from_numpy(inp[None, ...]).to(device)
            probs = net(inp_t)
            action, logp, entropy = sample_action(probs)
            action = action.squeeze(0).cpu().numpy()
            pattern_new = apply_action(pattern, action, cfg.pattern_min, cfg.pattern_max)
            dxy, dxz, score = evaluate_pattern(ctx, pattern_new)
            reward = 0.5 * ((dxy - prev_xy) + (dxz - prev_xz))
            baseline = baseline_decay * baseline + (1.0 - baseline_decay) * reward
            advantage = reward - baseline

            for _ in range(epochs_per_step):
                loss = -(logp.mean() * advantage) - entropy_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            # update pool
            if use_best_pool:
                pool.append(pattern_new)
                pool = sorted(pool, key=lambda p: evaluate_pattern(ctx, p)[2], reverse=True)[: cfg.rl['best_pool']]
            pattern = pattern_new
            prev_score = score
            prev_xy = dxy
            prev_xz = dxz
            if log_each_step:
                print(
                    f"[DPG] ep {ep+1}/{episodes} step {step+1}/{steps_per_episode} "
                    f"reward={reward:.6f} dxy={dxy:.6f} dxz={dxz:.6f} avg={score:.6f}",
                    flush=True,
                )

    return pattern, prev_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--mode", choices=["ga", "dpg", "dpg_best", "best"], default="ga")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode == "ga":
        best, score = run_ga(cfg)
        print("GA best", score)
    elif args.mode == "dpg":
        best, score = run_dpg(cfg, use_best_pool=False)
        print("DPG best", score)
    elif args.mode == "best":
        best, score = run_best(cfg)
        print("BEST hybrid", score)
    else:
        best, score = run_dpg(cfg, use_best_pool=True)
        print("DPG best-pool", score)
