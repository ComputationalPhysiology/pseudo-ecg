"""
Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3378475/
and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3075562/#R25
and https://opencarp.org/documentation/examples/02_ep_tissue/07_extracellular
"""

from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import cardiac_geometries
import ufl

import dolfin
import numpy as np
import cbcbeat

import demo_slab


def setup_conductivites(
    geo, chi, C_m, g_il=0.34, g_it=0.060, g_el=0.12, g_et=0.080, g_b=1.0
):

    A = demo_slab.define_fibers()

    factor = 1 / (chi * C_m)

    V_g = dolfin.FunctionSpace(geo.mesh, "DG", 0)

    bath_inds = np.where(geo.cfun.array() == geo.markers["Bath"][0])[0]
    cell_inds = np.where(geo.cfun.array() == geo.markers["Myocardium"][0])[0]

    g_il_fun = dolfin.Function(V_g)
    g_il_fun.vector()[cell_inds] = g_il * factor
    g_il_fun.vector()[bath_inds] = g_b * factor

    g_it_fun = dolfin.Function(V_g)
    g_it_fun.vector()[cell_inds] = g_it * factor
    g_it_fun.vector()[bath_inds] = g_b * factor

    g_el_fun = dolfin.Function(V_g)
    g_el_fun.vector()[cell_inds] = g_el * factor
    g_el_fun.vector()[bath_inds] = g_b * factor

    g_et_fun = dolfin.Function(V_g)
    g_et_fun.vector()[cell_inds] = g_et * factor
    g_et_fun.vector()[bath_inds] = g_b * factor

    M_e_star = ufl.diag(dolfin.as_vector([g_el_fun, g_et_fun, g_et_fun]))
    M_i_star = ufl.diag(dolfin.as_vector([g_il_fun, g_it_fun, g_it_fun]))

    G_e = A * M_e_star * A.T
    G_i = A * M_i_star * A.T

    return demo_slab.Conductivites(
        G_i=G_i,
        G_e=G_e,
        G_m=None,
        g_il=g_il * factor,
        g_it=g_it * factor,
        g_el=g_el * factor,
        g_et=g_et * factor,
        g_b=g_b * factor,
    )


def solve_bidomain(cardiac_model: cbcbeat.CardiacModel, xdmf_file: str):

    # Customize and create a splitting solver
    ps = cbcbeat.SplittingSolver.default_parameters()
    theta = 0.5
    dt = 0.1
    # Save every millisecond
    save_freq = int(np.round(1 / dt))
    scheme = "GRL1"

    ps["pde_solver"] = "bidomain"
    ps["theta"] = theta
    ps["enable_adjoint"] = False
    ps["BidomainSolver"]["linear_solver_type"] = "iterative"

    ps["apply_stimulus_current_to_pde"] = True
    ps["CardiacODESolver"]["scheme"] = scheme
    solver = cbcbeat.SplittingSolver(cardiac_model, params=ps)

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cardiac_model.cell_models().initial_conditions())

    # Time stepping parameters
    # dt = 0.1
    T = 500.0
    interval = (0.0, T)

    timer = dolfin.Timer("XXX Forward solve")  # Time the total solve

    # Set-up separate potential function for post processing
    VS0 = vs.function_space().sub(0)
    V = VS0.collapse()
    v = dolfin.Function(V)

    VUR0 = vur.function_space().sub(0)
    Ve = VUR0.collapse()
    ve = dolfin.Function(Ve)

    VUR1 = vur.function_space().sub(1)
    Ue = VUR1.collapse()
    ue = dolfin.Function(Ue)

    # Set-up object to optimize assignment from a function to subfunction
    assigner = dolfin.FunctionAssigner(V, VS0)
    assigner.assign(v, vs_.sub(0))

    assigner_ve = dolfin.FunctionAssigner(Ve, VUR0)
    assigner_ve.assign(ve, vur.sub(0))

    assigner_ue = dolfin.FunctionAssigner(Ue, VUR1)
    assigner_ue.assign(ue, vur.sub(1))

    # Solve!
    for i, (timestep, fields) in enumerate(solver.solve(interval, dt)):
        print(f"(i, t_0, t_1) = ({i}, {timestep})")

        if i % save_freq != 0:
            continue
        # Extract the components of the field (vs_ at previous timestep,
        # current vs, current vur)
        (vs_, vs, vur) = fields
        assigner.assign(v, vs.sub(0))
        assigner_ve.assign(ve, vur.sub(0))
        assigner_ue.assign(ue, vur.sub(1))

        with dolfin.XDMFFile(xdmf_file) as f:
            f.write_checkpoint(
                v,
                function_name="v",
                time_step=timestep[0],
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                ve,
                function_name="ve",
                time_step=timestep[0],
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )
            f.write_checkpoint(
                ue,
                function_name="ue",
                time_step=timestep[0],
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )

    timer.stop()


def main():

    geo = cardiac_geometries.geometry.Geometry.from_folder("slab-in-bath")
    heart = geo.mesh

    xdmf_file = "bidomain_slab.xdmf"
    figpath = "bidomain_ecg_slab.png"

    if not Path(xdmf_file).is_file():

        # Membrane capacitance
        C_m = 1.0  # mu F / cm^2
        chi = 2000  # cm^-1

        # Conductivies
        g_il = 0.34  # S / m
        g_it = 0.060  # S / m
        g_el = 0.12  # S / m
        g_et = 0.080  # S / m
        # Bath conductivit
        g_b = 1.0  # S / m

        conductivites = setup_conductivites(
            geo=geo,
            chi=chi,
            C_m=C_m,
            g_il=g_il,
            g_it=g_it,
            g_el=g_el,
            g_et=g_et,
            g_b=g_b,
        )

        time = dolfin.Constant(0.0)

        # For some reason with need a bit higher stimulus.
        stimulus = demo_slab.define_stimulus(heart, time, chi, C_m, amp=200000.0)
        cell_model = cbcbeat.Tentusscher_panfilov_2006_epi_cell()
        cardiac_model = cbcbeat.CardiacModel(
            heart, time, conductivites.G_i, conductivites.G_e, cell_model, stimulus
        )

        solve_bidomain(cardiac_model=cardiac_model, xdmf_file=xdmf_file)

    vurs = demo_slab.load_from_file(geo.mesh, xdmf_file, key="ue")

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
    for phie in vurs:
        for el in electrodes:
            ecg[el].append(phie(el))

    fig, ax = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for i, (p, u_e) in enumerate(ecg.items()):
        axi = ax[::-1].T.flatten()[i]
        axi.plot(u_e, label=str(p))
        axi.set_title(f"Electrode {i + 1}")
        axi.grid()

    fig.savefig(figpath)


if __name__ == "__main__":
    main()
