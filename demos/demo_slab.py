from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

import dolfin
import cbcbeat
import pseudo_ecg


dolfin.parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math"]
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 3
dolfin.parameters["form_compiler"]["representation"] = "uflacs"


def solve_monodomain(
    heart_mesh: dolfin.Mesh, xdmf_file: str, polynomial_degree: int = 1
):
    time = dolfin.Constant(0.0)

    # Define some external stimulus
    S1_marker = 1
    L = 0.5
    S1_subdomain = dolfin.CompiledSubDomain(
        "x[0] <= L + DOLFIN_EPS && x[1] <= L + DOLFIN_EPS",
        L=L,
    )
    S1_markers = dolfin.MeshFunction("size_t", heart_mesh, heart_mesh.topology().dim())
    S1_subdomain.mark(S1_markers, S1_marker)
    I_s = dolfin.Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=0.0,
        duration=5.0,
        amplitude=50.0,
        degree=0,
    )
    # Store input parameters in cardiac model
    stimulus = cbcbeat.Markerwise((I_s,), (1,), S1_markers)

    # Define the conductivity (tensors)
    M_i = 0.05
    # M_e = 1.0

    # Pick a cell model (see supported_cell_models for tested ones)
    cell_model = cbcbeat.Tentusscher_panfilov_2006_epi_cell()

    # Collect this information into the CardiacModel class
    cardiac_model = cbcbeat.CardiacModel(
        heart_mesh, time, M_i, None, cell_model, stimulus
    )

    # Customize and create a splitting solver
    ps = cbcbeat.SplittingSolver.default_parameters()
    theta = 0.5
    dt = 0.05
    scheme = "GRL1"
    preconditioner = "sor"

    ps["pde_solver"] = "monodomain"
    ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
    ps["MonodomainSolver"]["theta"] = theta
    ps["MonodomainSolver"]["polynomial_degree"] = polynomial_degree
    ps["MonodomainSolver"]["preconditioner"] = preconditioner
    ps["MonodomainSolver"]["default_timestep"] = dt
    ps["MonodomainSolver"]["use_custom_preconditioner"] = False
    ps["theta"] = theta
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True
    ps["CardiacODESolver"]["scheme"] = scheme
    ps["CardiacODESolver"]["polynomial_degree"] = polynomial_degree
    solver = cbcbeat.SplittingSolver(cardiac_model, params=ps)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cell_model.initial_conditions())

    # Time stepping parameters
    # dt = 0.1
    T = 50.0
    interval = (0.0, T)

    timer = dolfin.Timer("XXX Forward solve")  # Time the total solve

    # Set-up separate potential function for post processing
    VS0 = vs.function_space().sub(0)
    V = VS0.collapse()
    v = dolfin.Function(V)

    # Set-up object to optimize assignment from a function to subfunction
    assigner = dolfin.FunctionAssigner(V, VS0)
    assigner.assign(v, vs_.sub(0))

    # Solve!
    for timestep, fields in solver.solve(interval, dt):
        print("(t_0, t_1) = (%g, %g)", timestep)

        # Extract the components of the field (vs_ at previous timestep,
        # current vs, current vur)
        (vs_, vs, vur) = fields
        assigner.assign(v, vs.sub(0))
        with dolfin.XDMFFile(xdmf_file) as f:
            f.write_checkpoint(
                v,
                function_name="v",
                time_step=timestep[0],
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )

    timer.stop()


def load_voltage(heart_mesh, xdmf_file, polynomial_degree=1):
    V = dolfin.FunctionSpace(heart_mesh, "Lagrange", polynomial_degree)
    v = dolfin.Function(V)

    i = 0
    vs = []
    with dolfin.XDMFFile(xdmf_file) as f:
        while True:
            try:
                f.read_checkpoint(v, "v", i)
            except Exception:
                break
            else:
                vs.append(v.copy(deepcopy=True))
                i += 1
    return vs


def main():
    polynomial_degree = 2
    mesh = dolfin.RectangleMesh(
        dolfin.Point(-1.0, -1.0), dolfin.Point(2.0, 2.0), 100, 100
    )
    cfun = dolfin.MeshFunction("size_t", mesh, 2)

    class Heart(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return 0.0 - 3e-16 < x[0] < 1.0 + 3e-16 and 0.0 - 3e-16 < x[1] < 1.0 + 3e-16

    heart = Heart()
    cfun.set_all(0)
    heart.mark(cfun, 1)

    heart_mesh = pseudo_ecg.mesh_utils.extract_heart_from_torso(cfun, marker=1)

    xdmf_file = "v_cg2.xdmf"
    figpath = "ecg_cg2.png"

    if not Path(xdmf_file).is_file():
        solve_monodomain(heart_mesh, xdmf_file, polynomial_degree)

    vs = load_voltage(heart_mesh, xdmf_file, polynomial_degree=polynomial_degree)

    dx = dolfin.dx(subdomain_data=cfun, domain=mesh)(1)

    points = [
        (-1.0, 2.0),
        (2.0, 2.0),
        (-1.0, -1.0),
        (2.0, -1.0),
    ]
    dist = {}
    for point in points:
        dist[point] = pseudo_ecg.eikonal.distance(mesh, point=point, factor=15)

    ecg = defaultdict(list)
    for v in vs:
        for point in points:
            u_e = pseudo_ecg.ecg.ecg_recovery(
                v=v,
                mesh=mesh,
                sigma_i=0.02 * dolfin.Identity(2),
                r=dist[point],
                dx=dx,
                sigma_b=dolfin.Constant(1.0),
            )

            ecg[point].append(float(u_e))

    fig, ax = plt.subplots()
    for p, u_e in ecg.items():
        ax.plot(u_e, label=str(p))

    ax.legend()
    fig.savefig(figpath)


if __name__ == "__main__":
    main()
