import math
import dolfin

from pseudo_ecg.ecg import ecg_recovery


def test_ecg_recovery():
    mesh = dolfin.UnitCubeMesh(5, 5, 5)
    cfun = dolfin.MeshFunction("size_t", mesh, 3)

    class Heart(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return 0.25 < x[0] < 0.75 and 0.25 < x[1] < 0.75 and 0.25 < x[2] < 0.75

    heart = Heart()
    cfun.set_all(0)
    heart.mark(cfun, 1)
    dx = dolfin.dx(subdomain_data=cfun, domain=mesh)(1)

    # Need second order here, otherwise it will be zero
    V = dolfin.FunctionSpace(mesh, "CG", 2)
    v = dolfin.Function(V)
    v.interpolate(dolfin.Expression("100 * x[0] * x[0]", degree=2))

    u_e = ecg_recovery(
        v=v,
        mesh=mesh,
        sigma_i=dolfin.Identity(3),
        point=[1.0, 1.0, 1.0],
        dx=dx,
        sigma_b=dolfin.Constant(1.0),
    )

    # FIXE: Find some analytic expression
    assert math.isclose(float(u_e), 0.13810180497903637)


if __name__ == "__main__":
    test_ecg_recovery()
