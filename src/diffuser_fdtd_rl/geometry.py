from __future__ import annotations

import numpy as np


def make_diffuser_pattern(rng: np.random.Generator, shape=(10, 10), vmin=0, vmax=10) -> np.ndarray:
    return rng.integers(vmin, vmax + 1, size=shape, endpoint=True)


def apply_action(pattern: np.ndarray, action: np.ndarray, vmin=0, vmax=10) -> np.ndarray:
    # action shape = pattern shape, values in {-1,0,1}
    out = pattern + action
    out = np.clip(out, vmin, vmax)
    return out


def pattern_to_height_m(pattern: np.ndarray, max_depth_cm: float, cell_size_cm: float) -> np.ndarray:
    # pattern values indicate number of elements (0..10); each element = cell_size_cm
    return (pattern.astype(np.float32) * (cell_size_cm / 100.0))


def build_air_mask(
    grid_shape,
    pattern: np.ndarray,
    dx: float,
    cell_size_cm: float,
    diffuser_plane_x: int,
    center_yz: tuple[int, int],
):
    """
    Build air mask (1=air, 0=solid) for diffuser with wells extending toward -x.
    Pattern is mapped to voxels with segment size ~cell_size_cm.
    """
    nx, ny, nz = grid_shape
    air = np.ones(grid_shape, dtype=np.float32)
    seg_cells = max(1, int(round((cell_size_cm / 100.0) / dx)))
    H, W = pattern.shape
    total_y = H * seg_cells
    total_z = W * seg_cells
    cy, cz = center_yz
    y0 = max(0, cy - total_y // 2)
    z0 = max(0, cz - total_z // 2)
    y1 = min(ny, y0 + total_y)
    z1 = min(nz, z0 + total_z)

    # backing plate at diffuser_plane_x
    if 0 <= diffuser_plane_x < nx:
        air[diffuser_plane_x, y0:y1, z0:z1] = 0.0

    # wells extend toward -x (source side)
    for iy in range(H):
        for iz in range(W):
            depth_m = pattern[iy, iz] * (cell_size_cm / 100.0)
            depth_cells = int(round(depth_m / dx))
            if depth_cells <= 0:
                continue
            y_start = y0 + iy * seg_cells
            y_end = min(y1, y_start + seg_cells)
            z_start = z0 + iz * seg_cells
            z_end = min(z1, z_start + seg_cells)
            x_start = max(0, diffuser_plane_x - depth_cells)
            x_end = diffuser_plane_x
            air[x_start:x_end, y_start:y_end, z_start:z_end] = 0.0

    return air
