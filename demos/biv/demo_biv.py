from typing import NamedTuple
from pathlib import Path
from collections import defaultdict

import dolfin
import matplotlib.pyplot as plt
import cardiac_geometries
import numpy as np
import cbcbeat

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

import pseudo_ecg

dolfin.parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math"]
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 3
dolfin.parameters["form_compiler"]["representation"] = "uflacs"


def harmonic_mean(a, b):
    return a * b / (a + b)


def define_stimulus(mesh, time, chi, C_m, amp=500) -> cbcbeat.Markerwise:
    # Define some external stimulus
    S1_marker = 1
    S1_subdomain = dolfin.CompiledSubDomain("x[0] > 3.5")
    S1_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    S1_subdomain.mark(S1_markers, S1_marker)

    # amp = 500.0  # mu A/cm^3
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

    return cbcbeat.Markerwise((I_s,), (S1_marker,), S1_markers)


def load_from_file(heart_mesh, xdmffile, key="v"):
    V = dolfin.FunctionSpace(heart_mesh, "Lagrange", 1)
    v = dolfin.Function(V)

    i = 0
    vs = []
    with dolfin.XDMFFile(Path(xdmffile).as_posix()) as f:
        while True:
            try:
                f.read_checkpoint(v, key, i)
            except Exception:
                break
            else:
                vs.append(v.copy(deepcopy=True))
                i += 1
    return vs


class Conductivites(NamedTuple):
    G_i: ufl.tensors.ComponentTensor | None
    G_e: ufl.tensors.ComponentTensor | None
    G_m: ufl.tensors.ComponentTensor | None
    g_il: float
    g_it: float
    g_el: float
    g_et: float
    g_b: float


def setup_conductivites(
    chi,
    C_m,
    A,
    g_il=0.34,
    g_it=0.060,
    g_el=0.12,
    g_et=0.080,
    g_b=1.0,
    mesh=None,
    use_augmented_monodomain: bool = False,
    augmented_bath_size=0.01,
):
    g_bulk_l = harmonic_mean(g_il, g_el)
    g_bulk_t = harmonic_mean(g_it, g_et)

    factor = 1 / (chi * C_m)

    if use_augmented_monodomain:
        raise NotImplementedError
        g_edge_l = harmonic_mean(g_il, g_b)
        g_edge_t = harmonic_mean(g_it, g_b)

        bath_aug_marker = 1
        bath_aug_subdomain = dolfin.CompiledSubDomain(
            "x[2] < s + DOLFIN_EPS || x[2] > 0.5 - s - DOLFIN_EPS",
            s=augmented_bath_size,
        )
        bath_aug_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
        bath_aug_markers.set_all(0)
        bath_aug_subdomain.mark(bath_aug_markers, bath_aug_marker)

        edge_inds = np.where(bath_aug_markers.array() == 1)[0]
        bulk_inds = np.where(bath_aug_markers.array() == 0)[0]

        V_g = dolfin.FunctionSpace(mesh, "DG", 0)
        g_bulk_l_fun = dolfin.Function(V_g)
        g_bulk_l_fun.vector()[bulk_inds] = g_bulk_l * factor
        g_bulk_l_fun.vector()[edge_inds] = g_edge_l * factor

        g_bulk_t_fun = dolfin.Function(V_g)
        g_bulk_t_fun.vector()[bulk_inds] = g_bulk_t * factor
        g_bulk_t_fun.vector()[edge_inds] = g_edge_t * factor

        M_star = ufl.diag(dolfin.as_vector([g_bulk_l_fun, g_bulk_t_fun, g_bulk_t_fun]))

    else:
        M_star = ufl.diag(
            dolfin.as_vector(np.array([g_bulk_l, g_bulk_t, g_bulk_t]) * factor)
        )

    G_m = A * M_star * A.T

    M_e_star = ufl.diag(dolfin.as_vector(np.array([g_el, g_et, g_et]) * factor))
    M_i_star = ufl.diag(dolfin.as_vector(np.array([g_il, g_it, g_it]) * factor))
    G_e = A * M_e_star * A.T
    G_i = A * M_i_star * A.T

    return Conductivites(
        G_i=G_i,
        G_e=G_e,
        G_m=G_m,
        g_il=g_il * factor,
        g_it=g_it * factor,
        g_el=g_el * factor,
        g_et=g_et * factor,
        g_b=g_b * factor,
    )


def solve_monodomain(
    cardiac_model: cbcbeat.CardiacModel, xdmffile: Path, save_every_ms: float = 1
):
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
    vs_.assign(cardiac_model.cell_models().initial_conditions())

    # Time stepping parameters
    # dt = 0.1
    T = 500

    interval = (0.0, T)

    # timer = dolfin.Timer("XXX Forward solve")  # Time the total solve

    # Set-up separate potential function for post processing
    VS0 = vs.function_space().sub(0)
    V = VS0.collapse()
    v = dolfin.Function(V)

    # Set-up object to optimize assignment from a function to subfunction
    assigner = dolfin.FunctionAssigner(V, VS0)
    assigner.assign(v, vs_.sub(0))

    xdmffile = Path(xdmffile)
    xdmffile.unlink(missing_ok=True)
    xdmffile.with_suffix(".h5").unlink(missing_ok=True)

    # # Solve!
    for i, (timestep, fields) in enumerate(solver.solve(interval, dt)):
        print(f"(i, t_0, t_1) = ({i}, {timestep})")

        if i % save_freq != 0:
            continue

        (vs_, vs, vur) = fields
        assigner.assign(v, vs.sub(0))

        with dolfin.XDMFFile(xdmffile.as_posix()) as f:
            f.write_checkpoint(
                v,
                function_name="v",
                time_step=timestep[0],
                encoding=dolfin.XDMFFile.Encoding.HDF5,
                append=True,
            )


def compute_phie_recovery(
    electrodes: list[tuple[float, float, float]], mesh, g_b, vs: list[dolfin.Function]
) -> dict[tuple[float, float, float], list[float]]:
    ecg = defaultdict(list)
    print("Compute ecg recovert")
    for v in vs:
        for el in electrodes:
            ecg[el].append(
                pseudo_ecg.ecg.ecg_recovery(v=v, mesh=mesh, sigma_b=g_b, point=el)
            )
    return ecg


def main(
    use_augmented_monodomain=True,
    use_pseudobidomain=True,
    xdmffile="output.xdmf",
    figpath="slab_ecg_pba.png",
):
    xdmffile = Path(xdmffile)
    figpath = Path(figpath)

    geo = cardiac_geometries.geometry.Geometry.from_folder("biv")
    geo_torso = cardiac_geometries.geometry.Geometry.from_folder("biv-in-torso")

    heart = geo.mesh

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

    A = dolfin.as_matrix(
        [
            [geo.f0[0], geo.s0[0], geo.n0[0]],
            [geo.f0[1], geo.s0[1], geo.n0[1]],
            [geo.f0[2], geo.s0[2], geo.n0[2]],
        ]
    )

    conductivites = setup_conductivites(
        chi=chi,
        C_m=C_m,
        mesh=heart,
        A=A,
        augmented_bath_size=0.01,
        g_il=g_il,
        g_it=g_it,
        g_el=g_el,
        g_et=g_et,
        g_b=g_b,
        use_augmented_monodomain=use_augmented_monodomain,
    )

    time = dolfin.Constant(0.0)

    stimulus = define_stimulus(heart, time, chi, C_m)
    cell_model = cbcbeat.Tentusscher_panfilov_2006_epi_cell()
    cardiac_model = cbcbeat.CardiacModel(
        heart, time, conductivites.G_m, None, cell_model, stimulus
    )

    if not xdmffile.is_file():
        solve_monodomain(cardiac_model=cardiac_model, xdmffile=xdmffile)

    vs = load_from_file(heart, xdmffile, key="v")

    electrodes = [
        (4.0, 0.0, 0.0),  # ground
        (-1.0, 0.0, -2.0),  # L - left arm
        (-1.0, 0.0, 0.0),  # R - right arm
        (4.0, 0.0, -2.0),  # F - left leg
    ]

    # Indices
    left_arm_index = 1
    right_arm_index = 2
    left_leg_index = 3

    if use_pseudobidomain:
        A_torso = dolfin.as_matrix(
            [
                [geo_torso.f0[0], geo_torso.s0[0], geo_torso.n0[0]],
                [geo_torso.f0[1], geo_torso.s0[1], geo_torso.n0[1]],
                [geo_torso.f0[2], geo_torso.s0[2], geo_torso.n0[2]],
            ]
        )
        conductivites_torso = setup_conductivites(
            chi=chi,
            C_m=C_m,
            mesh=geo_torso.mesh,
            A=A_torso,
            augmented_bath_size=0.01,
            g_il=g_il,
            g_it=g_it,
            g_el=g_el,
            g_et=g_et,
            g_b=g_b,
            use_augmented_monodomain=use_augmented_monodomain,
        )
        phie = pseudo_ecg.ecg.pseudo_bidomain(
            mesh=geo_torso.mesh,
            cfun=geo_torso.cfun,
            heart_marker=geo_torso.markers["HEART"][0],
            torso_marker=geo_torso.markers["TISSUE"][0],
            G_i=conductivites_torso.G_i,
            G_e=conductivites_torso.G_e,
            g_b=conductivites_torso.g_b,
            vs=vs,
            electrodes=electrodes,
            xdmffile=Path(figpath).with_suffix(".xdmf"),
        )

        # breakpoint()

    else:
        # Use recovery
        phie = compute_phie_recovery(
            electrodes=electrodes, mesh=heart, g_b=conductivites.g_b, vs=vs
        )

    lead1 = np.subtract(
        phie[electrodes[left_arm_index]], phie[electrodes[right_arm_index]]
    )
    lead2 = np.subtract(
        phie[electrodes[left_leg_index]], phie[electrodes[right_arm_index]]
    )
    lead3 = np.subtract(
        phie[electrodes[left_leg_index]], phie[electrodes[left_arm_index]]
    )

    fig, ax = plt.subplots(1, 3, figsize=(12, 8), sharex=True, sharey=True)
    ax[0].plot(lead1)
    ax[0].set_title("Lead 1")

    ax[1].plot(lead2)
    ax[1].set_title("Lead 2")

    ax[2].plot(lead3)
    ax[2].set_title("Lead 3")

    for axi in ax.flatten():
        axi.grid()

    fig.savefig(figpath)
    np.save(Path(figpath).with_suffix(".npy"), phie, allow_pickle=True)


if __name__ == "__main__":
    for use_augmented_monodomain, use_pseudobidomain, xdmffile, figpath in [
        # (False, False, "out_mono.xdmf", "mono.png"),
        (False, True, "out_mono.xdmf", "pb_mono.png"),
        # (True, False, "out_aug_mono.xdmf", "aug_mono.png"),
        # (True, True, "out_aug_mono.xdmf", "pb_aug_mono.png"),
    ]:
        main(
            use_augmented_monodomain=use_augmented_monodomain,
            use_pseudobidomain=use_pseudobidomain,
            xdmffile=xdmffile,
            figpath=figpath,
        )
