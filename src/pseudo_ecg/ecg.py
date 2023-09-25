from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import numpy as np
import dolfin

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from .mesh_utils import vertex_map_kdtree


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


def pseudo_bidomain(
    mesh: dolfin.Mesh,
    cfun: dolfin.MeshFunction,
    heart_marker: int,
    torso_marker: int,
    G_i: ufl.tensors.ComponentTensor,
    G_e: ufl.tensors.ComponentTensor,
    g_b: float,
    vs: list[dolfin.Function],
    electrodes: list[tuple[float, float, float]],
    xdmffile: Path | str | None = None,
):
    V = dolfin.FunctionSpace(mesh, "P", 1)
    VmT = dolfin.Function(V, name="Vm_torso")
    ue = dolfin.Function(V, name="ue")

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dxx = ufl.Measure("dx", domain=mesh, subdomain_data=cfun)

    Atorso = ufl.inner((G_i + G_e) * ufl.grad(u), ufl.grad(v)) * dxx(
        heart_marker
    ) + dolfin.Constant(g_b) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dxx(torso_marker)

    Ltorso = -ufl.inner(G_i * ufl.grad(VmT), ufl.grad(v)) * dxx(heart_marker)

    At = dolfin.assemble(Atorso)
    bt = dolfin.assemble(Ltorso)
    solverT = dolfin.PETScKrylovSolver("cg")
    solverT.set_operator(At)

    # nullspace for torso problem
    v0 = dolfin.Vector(VmT.vector())
    V.dofmap().set(v0, 1.0)
    v0 *= 1.0 / v0.norm("l2")
    null_space = dolfin.VectorSpaceBasis([v0])
    dolfin.as_backend_type(At).set_nullspace(null_space)

    Vh = vs[0].function_space()
    vmap = vertex_map_kdtree(Vh.mesh(), mesh)

    if xdmffile is not None:
        xdmffile = Path(xdmffile)
        xdmffile.unlink(missing_ok=True)
        xdmffile.with_suffix(".h5").unlink(missing_ok=True)

    ecg = defaultdict(list)
    for i, vh in enumerate(vs):
        VmT.vector()[dolfin.vertex_to_dof_map(V)[vmap]] = vh.vector()[
            dolfin.vertex_to_dof_map(Vh)
        ].copy()
        dolfin.assemble(Ltorso, tensor=bt)
        null_space.orthogonalize(bt)
        solverT.solve(ue.vector(), bt)

        if xdmffile is not None:
            with dolfin.XDMFFile(xdmffile.as_posix()) as f:
                f.write_checkpoint(
                    ue,
                    function_name="ue",
                    time_step=i,
                    encoding=dolfin.XDMFFile.Encoding.HDF5,
                    append=True,
                )

        for el in electrodes:
            ecg[el].append(ue(el))

    return ecg
