from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


def db(message: str):
    # Minimal debug print hook (can be wired to logging later)
    print(message)


def speed_of_sound(temp_c: float) -> float:
    # Approximation (m/s)
    return 331.3 + 0.606 * temp_c


def air_density(temp_c: float, pressure_hpa: float) -> float:
    # Ideal gas approx (kg/m^3)
    T = temp_c + 273.15
    p = pressure_hpa * 100.0
    R = 287.05
    return p / (R * T)


def check_courant(courant: float) -> None:
    # Allow a tiny epsilon to avoid false positives from floating-point rounding.
    limit = 1.0 / math.sqrt(3.0)
    if courant > limit + 1e-9:
        raise ValueError("Courant exceeds 1/sqrt(3) stability limit for 3D FDTD.")


def hertz_to_omega(f: float) -> float:
    return 2.0 * math.pi * f


def gaussian_pulse(t: float, fmax: float) -> float:
    # Band-limited Gaussian pulse (simple approximation)
    # Based on standard pulse used in acoustic FDTD literature
    # Use t0 so that pulse is centered; sigma chosen from fmax
    sigma = 1.0 / (2.0 * math.pi * fmax)
    t0 = 6 * sigma
    return math.exp(-((t - t0) ** 2) / (2 * sigma**2))


@dataclass
class GridSpec:
    nx: int
    ny: int
    nz: int
    dx: float
    dt: float

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)
