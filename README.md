# DiffuserWallOptimization

End‑to‑end simulation and optimization of Schroeder diffuser wall patterns using FDTD acoustic modeling and reinforcement learning, designed to replicate the methodology from the paper **“Reinforcement Learning Algorithm and FDTD‑based Simulation Applied to Schroeder Diffuser Design Optimization.”**

The project includes:
- 3D acoustic wave propagation using FDTD with lossy terms and PML boundaries
- Schroeder diffuser geometry construction (10×10 pattern, 0–10 depth units)
- Polar response extraction + diffusion coefficient calculation
- Optimization methods:
  - Genetic Algorithm (GA)
  - Deep Policy Gradient (DPG)
  - Hybrid **GA → DPG (best‑pool refinement)**
- Apple GPU (MPS) acceleration via PyTorch

---

## Project Layout

```
DiffuserWallOptimization/
├─ src/diffuser_fdtd_rl/     # core implementation (FDTD, RL, GA, metrics)
├─ configs/
│  ├─ paper.yaml             # paper‑scale config (slow, accurate)
│  └─ smoke.yaml             # small config for quick testing
├─ pdf_pages/                # rendered paper pages
├─ Reinforcement_Learning_Algorithm_and_FDTD-Based_Si.pdf
├─ Reinforcement_Learning_Algorithm_and_FDTD-Based_Si.md
└─ README.md
```

---

## Requirements

- Python 3.10+
- PyTorch with MPS support (Apple Silicon GPU)
- NumPy, SciPy, Matplotlib

If using your **system Python** (not a venv):

```
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib
```

Verify MPS support:

```
python - <<'PY'
import torch
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())
PY
```

---

## Quick Smoke Test (recommended first)

Runs on a tiny grid to verify that everything works.

```
PYTHONPATH=src python -m diffuser_fdtd_rl.experiment --config configs/smoke.yaml --mode ga
PYTHONPATH=src python -m diffuser_fdtd_rl.experiment --config configs/smoke.yaml --mode dpg
PYTHONPATH=src python -m diffuser_fdtd_rl.experiment --config configs/smoke.yaml --mode best
```

Each run logs progress to terminal.

---

## Paper‑Scale Optimization (Apple GPU)

This is **very slow** (hours) but matches the scale and parameters from the paper.

```
PYTHONPATH=src python -m diffuser_fdtd_rl.experiment --config configs/paper.yaml --mode ga
PYTHONPATH=src python -m diffuser_fdtd_rl.experiment --config configs/paper.yaml --mode dpg
PYTHONPATH=src python -m diffuser_fdtd_rl.experiment --config configs/paper.yaml --mode best
```

Modes:
- `ga` = Genetic Algorithm
- `dpg` = Deep Policy Gradient
- `dpg_best` = DPG with best‑pool retention
- `best` = Hybrid GA → DPG (highest quality)

---

## Key Implementation Notes

- **Courant stability**: timestep is derived from Courant (`sqrt(1/3)`), which may slightly differ from the nominal sampling rate; this is intentional for stability.
- **Feature inputs to DPG**: pattern, FFT2, cepstrum, and autocorrelation resized with `RectBivariateSpline` (order 5), per paper.
- **Reward**: based on improvements in XY and YZ diffusion coefficients.
- **Caching**: simulation outputs are cached per pattern to speed up training.

---

## Configs

Edit `configs/paper.yaml` to adjust:
- Training budget (`episodes`, `steps_per_episode`, `epochs_per_step`)
- GA population + generations
- Device (`mps` or `cpu`)

---

## Suggested Workflow

1. Run smoke tests
2. Run `best` mode on `configs/paper.yaml`
3. Tune RL/GA parameters in `configs/paper.yaml`

---

## License

This project is for research/replication use and is based on a published paper. Please cite the paper when using or extending this work.
