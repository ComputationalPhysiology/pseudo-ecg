from scipy.spatial import KDTree
import dolfin
import numpy as np
import numpy.typing as npt


def facet_function_from_heart_mesh(
    ffun: dolfin.MeshFunction,
    heart_mesh: dolfin.Mesh,
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


def vertex_map_kdtree(
    submesh: dolfin.Mesh, mesh: dolfin.Mesh
) -> npt.NDArray[np.uint64]:
    """Get vertex map between submesh and mesh.

    Parameters
    ----------
    submesh : dolfin.Mesh
        A submesh of `mesh` (for example a heart)
    mesh : dolfin.Mesh
        A full mesh (for example a heart with torso)

    Returns
    -------
    npt.NDArray[np.uint64]
        List of indices in the parent mesh. For example the first
        value in the list will correspond to the first vertex in the
        submesh and the value will be the corresponding index in the
        mesh

    Note
    ----
    This is useful if for example you have a submesh which is a heart and a mesh
    which is a heart with a torso and you want to find the indices
    in the torso mesh that corresponds to the indices in the heart mesh.
    """

    # Build KDTree
    tree = KDTree([v.point().array() for v in dolfin.vertices(mesh)])

    # Allocate array for vertex map
    vmap = np.empty(submesh.num_vertices(), dtype="uint64")
    # Loop over all vertices in submesh and set the closest index
    # in the parent mesh
    for i, vertex in enumerate(dolfin.vertices(submesh)):
        vmap[i] = tree.query(vertex.point().array())[1]

    return vmap


def vertex_map_meshview(
    submesh: dolfin.Mesh, mesh: dolfin.Mesh
) -> npt.NDArray[np.uint64]:
    """Get vertex map between submesh and mesh when the submesh is a `dolfin.MeshView`.

    Parameters
    ----------
    submesh : dolfin.Mesh
        A submesh of `mesh` (for example a heart)
    mesh : dolfin.Mesh
        A full mesh (for example a heart with torso)

    Returns
    -------
    npt.NDArray[np.uint64]
        List of indices in the parent mesh. For example the first
        value in the list will correspond to the first vertex in the
        submesh and the value will be the corresponding index in the
        mesh

    Note
    ----
    This is useful if for example you have a submesh which is a heart and a mesh
    which is a heart with a torso and you want to find the indices
    in the torso mesh that corresponds to the indices in the heart mesh.
    """

    return np.array(
        submesh.topology().mapping()[mesh.id()].vertex_map(), dtype="uint64"
    )


def vertex_map_submesh(
    submesh: dolfin.Mesh, mesh: dolfin.Mesh
) -> npt.NDArray[np.uint64]:
    """Get vertex map between submesh and mesh when the submesh is a `dolfin.SubMesh`.

    Parameters
    ----------
    submesh : dolfin.Mesh
        A submesh of `mesh` (for example a heart)
    mesh : dolfin.Mesh
        A full mesh (for example a heart with torso)

    Returns
    -------
    npt.NDArray[np.uint64]
        List of indices in the parent mesh. For example the first
        value in the list will correspond to the first vertex in the
        submesh and the value will be the corresponding index in the
        mesh

    Note
    ----
    This is useful if for example you have a submesh which is a heart and a mesh
    which is a heart with a torso and you want to find the indices
    in the torso mesh that corresponds to the indices in the heart mesh.
    """

    return submesh.data().array("parent_vertex_indices", 0)


def surface_to_volume_ratio(mesh: dolfin.Mesh) -> float:
    """Compute the surface area divided by the volume"""
    area = dolfin.assemble(dolfin.Constant(1.0) * dolfin.ds(domain=mesh))
    volume = dolfin.assemble(dolfin.Constant(1.0) * dolfin.dx(domain=mesh))
    return area / volume
