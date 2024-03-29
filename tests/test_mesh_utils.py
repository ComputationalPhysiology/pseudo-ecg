from typing import NamedTuple

import dolfin
import pytest
import numpy as np

from pseudo_ecg import mesh_utils


class Data(NamedTuple):
    mesh: dolfin.Mesh
    cfun: dolfin.MeshFunction
    marker: int


@pytest.fixture(scope="session")
def data():
    mesh = dolfin.UnitCubeMesh(3, 3, 3)
    cfun = dolfin.MeshFunction("size_t", mesh, 3)

    class Heart(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return 0.32 < x[0] < 0.67 and 0.32 < x[1] < 0.67 and 0.32 < x[2] < 0.67

    heart = Heart()
    cfun.set_all(0)
    heart.mark(cfun, 1)
    return Data(mesh=mesh, cfun=cfun, marker=1)


def test_extract_heart_from_torso(data: Data):
    heart_mesh = dolfin.MeshView.create(data.cfun, data.marker)
    assert (0.32 < heart_mesh.coordinates()).all()
    assert (heart_mesh.coordinates() < 0.67).all()


def test_facet_function_from_heart_mesh(data: Data):
    ffun = dolfin.MeshFunction("size_t", data.mesh, 2)
    outer_sides = dolfin.CompiledSubDomain("on_boundary")
    x0_side = dolfin.CompiledSubDomain(
        "near(x[0], 0.33, tol) && "
        "x[1] > 0.32 && x[1] < 0.67 && "
        "x[2] > 0.32 && x[2] < 0.67",
        tol=0.05,
    )

    x1_side = dolfin.CompiledSubDomain(
        "near(x[0], 0.66, tol) && "
        "x[1] > 0.32 && x[1] < 0.67 && "
        "x[2] > 0.32 && x[2] < 0.67",
        tol=0.05,
    )

    ffun.set_all(2)
    outer_sides.mark(ffun, 3)
    x0_side.mark(ffun, 4)
    x1_side.mark(ffun, 5)

    heart_mesh = dolfin.MeshView.create(data.cfun, data.marker)
    heart_ffun = mesh_utils.facet_function_from_heart_mesh(
        ffun=ffun, heart_mesh=heart_mesh
    )

    # Make sure we have a facet function on the heart mesh
    assert heart_ffun.size() == heart_mesh.num_facets()

    # Should be equally many marked facets
    assert (heart_ffun.array() == 4).sum() == (ffun.array() == 4).sum()
    assert (heart_ffun.array() == 5).sum() == (ffun.array() == 5).sum()


def test_vertex_map_meshview(data: Data):
    heart_mesh = dolfin.MeshView.create(data.cfun, data.marker)
    vmap = mesh_utils.vertex_map_meshview(heart_mesh, data.mesh)
    vmap_tree = mesh_utils.vertex_map_kdtree(heart_mesh, data.mesh)
    assert np.allclose(vmap, vmap_tree)


def test_vertex_map_submesh(data: Data):
    heart_mesh = dolfin.SubMesh(data.mesh, data.cfun, data.marker)
    vmap = mesh_utils.vertex_map_submesh(heart_mesh, data.mesh)
    vmap_tree = mesh_utils.vertex_map_kdtree(heart_mesh, data.mesh)
    assert np.allclose(vmap, vmap_tree)


def test_surface_to_volume_ratio():
    mesh = dolfin.BoxMesh(
        dolfin.MPI.comm_world,
        dolfin.Point(0.0, 0.0, 0.0),
        dolfin.Point(20, 7, 3),
        3,
        3,
        3,
    )
    chi = mesh_utils.surface_to_volume_ratio(mesh)
    assert np.isclose(chi, (2 * (20 * 7 + 20 * 3 + 7 * 3) / (20 * 7 * 3)))
