import dolfin
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl


def get_point_source(point: np.ndarray) -> dolfin.SubDomain:
    class PointSource3D(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return (
                dolfin.near(x[0], point[0])
                and dolfin.near(x[1], point[1])
                and dolfin.near(x[2], point[2])
            )

    class PointSource2D(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[0], point[0]) and dolfin.near(x[1], point[1])

    n = len(point)
    if n == 2:
        return PointSource2D()

    elif n == 3:
        return PointSource3D()

    else:
        raise ValueError(f"Expected point to be of dimension 2 or 3, got {n}")


def distance(
    mesh: dolfin.Mesh, point: np.ndarray, factor: float = 25
) -> dolfin.Function:
    P = dolfin.FiniteElement("P", mesh.ufl_cell(), 1)
    V = dolfin.FunctionSpace(mesh, P)
    dx = ufl.dx(domain=mesh)
    v = dolfin.TestFunction(V)
    u = dolfin.TrialFunction(V)
    dist = dolfin.Function(V)

    bc = dolfin.DirichletBC(V, 0, get_point_source(point=point), "pointwise")

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
    L = v * dolfin.Constant(1) * dx
    dolfin.solve(a == L, dist, bc, solver_parameters={"linear_solver": "lu"})

    # Create Eikonal problem
    eps = dolfin.Constant(mesh.hmax() / factor)
    F = (
        ufl.sqrt(dolfin.inner(ufl.grad(dist), ufl.grad(dist))) * v * dx
        - dolfin.Constant(1.0) * v * dx
        + eps * dolfin.inner(ufl.grad(dist), ufl.grad(v)) * dx
    )
    dolfin.solve(
        F == 0,
        dist,
        bc,
        solver_parameters={
            "newton_solver": {"linear_solver": "lu", "relative_tolerance": 1e-5},
        },
    )

    return dist
