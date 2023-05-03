from __future__ import annotations

import numpy as np
import dolfin
import ufl


from .eikonal import distance


def ecg_recovery(
    v: dolfin.Function,
    mesh: dolfin.Mesh,
    sigma_i,
    point: np.ndarray,
    dx: dolfin.Measure,
    sigma_b: dolfin.Constant,
):
    r = distance(mesh, point=point)
    grad_vm = ufl.grad(v)
    int_heart_expr = (ufl.div(sigma_i * grad_vm) / r) * dx
    int_heart = dolfin.assemble(int_heart_expr)
    return 1 / (4 * ufl.pi * sigma_b) * int_heart
