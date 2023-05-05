from typing import List
from scipy.spatial import KDTree
import dolfin


def extract_heart_from_torso(cfun: dolfin.MeshFunction, marker: int) -> dolfin.Mesh:
    return dolfin.MeshView.create(cfun, marker)


def facet_function_from_heart_mesh(
    ffun: dolfin.MeshFunction, heart_mesh: dolfin.Mesh, markers: List[int]
) -> dolfin.MeshFunction:
    assert ffun.mesh().id() in heart_mesh.topology().mapping()

    # Create a KDTree for the facet midpoints
    tree = KDTree([f.midpoint().array() for f in dolfin.facets(ffun.mesh())])

    # New facet function
    D = ffun.mesh().topology().dim()
    new_ffun = dolfin.MeshFunction("size_t", heart_mesh, D - 1, 0)

    for heart_cell in dolfin.cells(heart_mesh):
        for facet in dolfin.facets(heart_cell):
            global_index = tree.query(facet.midpoint().array())[1]
            new_ffun[facet] = ffun[global_index]

    return new_ffun
