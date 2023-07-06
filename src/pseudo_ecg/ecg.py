from __future__ import annotations

import numpy as np
import dolfin
import ufl


def ecg_recovery(
    v: dolfin.Function,
    mesh: dolfin.Mesh,
    sigma_i,
    dx: dolfin.Measure,
    sigma_b: dolfin.Constant,
    point: np.ndarray | None = None,
    eikonal_factor: float = 25.0,
    r: dolfin.Function | None = None,
):
    if r is None:
        r = dolfin.sqrt((dolfin.SpatialCoordinate(mesh) - dolfin.Constant(point)) ** 2)

    int_heart_expr = (ufl.div(sigma_i * ufl.grad(v)) / r) * dx
    int_heart = dolfin.assemble(int_heart_expr)
    return 1 / (4 * ufl.pi * sigma_b) * int_heart
