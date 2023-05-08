from __future__ import annotations

import numpy as np
import dolfin
import ufl


from .eikonal import distance


def ecg_recovery(
    v: dolfin.Function,
    mesh: dolfin.Mesh,
    sigma_i,
    dx: dolfin.Measure,
    sigma_b: dolfin.Constant,
    function_space: dolfin.FunctionSpace | None = None,
    point: np.ndarray | None = None,
    eikonal_factor: float = 25.0,
    r: dolfin.Function | None = None,
):
    if r is None:
        r = distance(mesh, point=point, factor=eikonal_factor)

    expr = sigma_i * ufl.grad(v)
    if function_space is not None:
        expr = dolfin.project(expr, function_space)

    int_heart_expr = (ufl.nabla_div(expr) / r) * dx
    int_heart = dolfin.assemble(int_heart_expr)
    return 1 / (4 * ufl.pi * sigma_b) * int_heart
