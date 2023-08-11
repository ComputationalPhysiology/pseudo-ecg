"""
Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3378475/
and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3075562/#R25
and https://opencarp.org/documentation/examples/02_ep_tissue/07_extracellular
"""

from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import cardiac_geometries

import dolfin
import ufl
import numpy as np
import cbcbeat
import pseudo_ecg


dolfin.parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math"]
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 3
dolfin.parameters["form_compiler"]["representation"] = "uflacs"


def solve_monodomain(geo, xdmf_file: str, chi: float, C_m: float):
    time = dolfin.Constant(0.0)

    # Define some external stimulus
    S1_marker = 1
    S1_subdomain = dolfin.CompiledSubDomain("x[0] <= 0.01")
    S1_markers = dolfin.MeshFunction("size_t", geo.mesh, geo.mesh.topology().dim())
    S1_subdomain.mark(S1_markers, S1_marker)

    # Pick a cell model (see supported_cell_models for tested ones)
    cell_model = cbcbeat.Tentusscher_panfilov_2006_epi_cell()

    amp = 50000.0  # mu A/cm^3
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention

    amplitude = factor * amp  # mV/ms
    I_s = dolfin.Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=1.0,
        duration=1.0,
        amplitude=amplitude,
        degree=0,
    )

    stimulus = cbcbeat.Markerwise((I_s,), (S1_marker,), S1_markers)

    # Define the conductivity (tensors)
    fiber = dolfin.as_vector([1, 0, 0])
    sheet = dolfin.as_vector([0, 1, 0])
    cross_sheet = dolfin.as_vector([0, 0, 1])

    A = dolfin.as_matrix(
        [
            [fiber[0], sheet[0], cross_sheet[0]],
            [fiber[1], sheet[1], cross_sheet[1]],
            [fiber[2], sheet[2], cross_sheet[2]],
        ]
    )
    from ufl import diag

    g_il = 0.34  # S / m
    g_it = 0.060  # S / m
    g_el = 0.12  # S / m
    g_et = 0.080  # S / m

    def harmonic_mean(a, b):
        return a * b / (a + b)

    g_bulk_l = harmonic_mean(g_il, g_el) * factor
    g_bulk_t = harmonic_mean(g_it, g_et) * factor

    M_star = diag(dolfin.as_vector([g_bulk_l, g_bulk_t, g_bulk_t]))
    M = A * M_star * A.T

    # Collect this information into the CardiacModel class
    cardiac_model = cbcbeat.CardiacModel(geo.mesh, time, M, None, cell_model, stimulus)

    # Customize and create a splitting solver
    ps = cbcbeat.SplittingSolver.default_parameters()
    theta = 0.5
    dt = 0.1
    # Save every millisecond
    save_freq = int(np.round(1 / dt))
    scheme = "GRL1"

    ps["pde_solver"] = "monodomain"
    ps["theta"] = theta
    ps["apply_stimulus_current_to_pde"] = True
    ps["CardiacODESolver"]["scheme"] = scheme
    solver = cbcbeat.SplittingSolver(cardiac_model, params=ps)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cell_model.initial_conditions())

    # Time stepping parameters
    # dt = 0.1
    T = 500
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
    for i, (timestep, fields) in enumerate(solver.solve(interval, dt)):
        print(f"(i, t_0, t_1) = ({i}, {timestep})")

        if i % save_freq != 0:
            continue

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


def load_from_file(heart_mesh, xdmf_file, key):
    V = dolfin.FunctionSpace(heart_mesh, "Lagrange", 1)
    v = dolfin.Function(V)

    i = 0
    vs = []
    with dolfin.XDMFFile(xdmf_file) as f:
        while True:
            try:
                f.read_checkpoint(v, key, i)
            except Exception:
                break
            else:
                vs.append(v.copy(deepcopy=True))
                i += 1
    return vs


def load_voltage(heart_mesh, xdmf_file):
    return load_from_file(heart_mesh=heart_mesh, xdmf_file=xdmf_file, key="v")


def surface_to_volume_ratio(mesh):
    # Surface to volume ratio
    area = dolfin.assemble(dolfin.Constant(1.0) * dolfin.ds(domain=mesh))
    volume = dolfin.assemble(dolfin.Constant(1.0) * dolfin.dx(domain=mesh))
    return area / volume  # cm^{-1}


def pseudo_bidomain():
    xdmf_file = "monodomain_slab.xdmf"
    geo_slab = cardiac_geometries.geometry.Geometry.from_folder("slab")
    vs = load_voltage(geo_slab.mesh, xdmf_file)

    geo = cardiac_geometries.geometry.Geometry.from_folder("slab-in-bath")
    heart = dolfin.MeshView.create(geo.cfun, geo.markers["Myocardium"][0])

    g_il = 0.34  # S / m
    g_it = 0.060  # S / m
    g_el = 0.12  # S / m
    g_et = 0.080  # S / m
    g_b = 1.0

    V_g = dolfin.FunctionSpace(geo.mesh, "DG", 0)

    fiber = dolfin.as_vector([1, 0, 0])
    sheet = dolfin.as_vector([0, 1, 0])
    cross_sheet = dolfin.as_vector([0, 0, 1])

    A = dolfin.as_matrix(
        [
            [fiber[0], sheet[0], cross_sheet[0]],
            [fiber[1], sheet[1], cross_sheet[1]],
            [fiber[2], sheet[2], cross_sheet[2]],
        ]
    )

    bath_inds = np.where(geo.cfun.array() == geo.markers["Bath"][0])[0]
    cell_inds = np.where(geo.cfun.array() == geo.markers["Myocardium"][0])[0]

    g_il_fun = dolfin.Function(V_g)
    g_il_fun.vector()[cell_inds] = g_il
    g_il_fun.vector()[bath_inds] = g_b

    g_it_fun = dolfin.Function(V_g)
    g_it_fun.vector()[cell_inds] = g_it
    g_it_fun.vector()[bath_inds] = g_b

    g_el_fun = dolfin.Function(V_g)
    g_el_fun.vector()[cell_inds] = g_el
    g_el_fun.vector()[bath_inds] = g_b

    g_et_fun = dolfin.Function(V_g)
    g_et_fun.vector()[cell_inds] = g_et
    g_et_fun.vector()[bath_inds] = g_b

    M_e_star = ufl.diag(dolfin.as_vector([g_el_fun, g_et_fun, g_et_fun]))
    M_i_star = ufl.diag(dolfin.as_vector([g_il_fun, g_it_fun, g_it_fun]))
    M_e = A * M_e_star * A.T
    M_i = A * M_i_star * A.T

    V = dolfin.FunctionSpace(geo.mesh, "P", 1)
    phie = dolfin.Function(V)
    w = dolfin.TestFunction(V)

    V_heart = dolfin.FunctionSpace(heart, "P", 1)
    v_heart = dolfin.Function(V_heart)

    for i, v in enumerate(vs):

        v_heart.interpolate(v)
        G = ((M_i + M_e) * ufl.grad(phie) + M_i * ufl.grad(v_heart)) * ufl.grad(w)
        # Define variational problem
        a, L = dolfin.system(G)
        pde = dolfin.LinearVariationalProblem(a, L, phie)
        solver = dolfin.LinearVariationalSolver(pde)
        solver.solve()

        with dolfin.XDMFFile("pseudo_bidomain.xdmf") as xdmf:
            xdmf.write_checkpoint(
                phie,
                function_name="phie",
                time_step=i,
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )


def main():

    geo = cardiac_geometries.geometry.Geometry.from_folder("slab")
    xdmf_file = "monodomain_slab.xdmf"
    figpath = "monodomain_ecg_slab.png"

    # Membrane capacitance
    C_m = 1.0  # mu F / cm^2
    chi = surface_to_volume_ratio(geo.mesh)

    if not Path(xdmf_file).is_file():
        solve_monodomain(geo, xdmf_file, C_m=C_m, chi=chi)

    vs = load_voltage(geo.mesh, xdmf_file)
    g_b = 1.0 / (C_m * chi)

    electrodes = [
        (0.0, 0.0, -0.1),
        (0.5, 0.0, -0.1),
        (1.0, 0.0, -0.1),
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.25),
        (0.5, 0.0, 0.25),
        (1.0, 0.0, 0.25),
    ]

    ecg = defaultdict(list)
    for i, v in enumerate(vs):
        print(i)
        for el in electrodes:
            ecg[el].append(
                pseudo_ecg.ecg.ecg_recovery(v=v, mesh=geo.mesh, sigma_b=g_b, point=el)
            )

    fig, ax = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for i, (p, u_e) in enumerate(ecg.items()):
        axi = ax[::-1].T.flatten()[i]
        axi.plot(u_e, label=str(p))
        axi.set_title(f"Electrode {i + 1}")
        axi.grid()

    fig.savefig(figpath)


if __name__ == "__main__":
    # main()
    pseudo_bidomain()
