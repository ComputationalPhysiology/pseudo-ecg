from __future__ import annotations

import numpy as np
import dolfin
import ufl


def ecg_recovery(
    *,
    v: dolfin.Function,
    mesh: dolfin.Mesh,
    sigma_b: dolfin.Constant,
    dx: dolfin.Measure | None = None,
    point: np.ndarray | None = None,
    r: dolfin.Function | None = None,
):
    if dx is None:
        dx = dolfin.dx(domain=mesh)
    if r is None:
        r = dolfin.SpatialCoordinate(mesh) - dolfin.Constant(point)
    r3 = ufl.sqrt((r**2)) ** 3
    return (1 / (4 * ufl.pi * sigma_b)) * dolfin.assemble(
        (ufl.inner(ufl.grad(v), r) / r3) * dx
    )
