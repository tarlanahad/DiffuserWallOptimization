from __future__ import annotations

import numpy as np


def pml_alpha_profile(shape, thickness, alpha_max, order=2.0):
    """
    Create a 3D alpha profile for PML. Alpha increases from 0 to alpha_max
    over the PML thickness on each boundary. The paper uses |(x-x0)/(xmax-x0)|^n.
    """
    nx, ny, nz = shape
    alpha = np.zeros(shape, dtype=np.float32)

    def axis_profile(n):
        prof = np.zeros(n, dtype=np.float32)
        for i in range(n):
            dist = 0
            if i < thickness:
                dist = thickness - i
            elif i >= n - thickness:
                dist = i - (n - thickness - 1)
            if dist > 0:
                x = dist / thickness
                prof[i] = alpha_max * (x**order)
        return prof

    px = axis_profile(nx)
    py = axis_profile(ny)
    pz = axis_profile(nz)

    # combine (max) across axes using broadcasting
    alpha = np.maximum(px[:, None, None], np.maximum(py[None, :, None], pz[None, None, :])).astype(np.float32)

    return alpha
