import dolfin

from pseudo_ecg import mesh_utils


def test_extract_heart_from_torso():
    mesh = dolfin.UnitCubeMesh(3, 3, 3)
    cfun = dolfin.MeshFunction("size_t", mesh, 3)

    class Heart(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return 0.32 < x[0] < 0.67 and 0.32 < x[1] < 0.67 and 0.32 < x[2] < 0.67

    heart = Heart()
    cfun.set_all(0)
    heart.mark(cfun, 1)

    heart_mesh = mesh_utils.extract_heart_from_torso(cfun, marker=1)
    assert (0.32 < heart_mesh.coordinates()).all()
    assert (heart_mesh.coordinates() < 0.67).all()


def test_facet_function_from_heart_mesh():
    mesh = dolfin.UnitCubeMesh(3, 3, 3)
    cfun = dolfin.MeshFunction("size_t", mesh, 3)

    class Heart(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return 0.32 < x[0] < 0.67 and 0.32 < x[1] < 0.67 and 0.32 < x[2] < 0.67

    heart = Heart()
    cfun.set_all(0)
    heart.mark(cfun, 1)

    ffun = dolfin.MeshFunction("size_t", mesh, 2)
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

    heart_mesh = mesh_utils.extract_heart_from_torso(cfun, marker=1)
    heart_ffun = mesh_utils.facet_function_from_heart_mesh(
        ffun=ffun, heart_mesh=heart_mesh, markers=[4, 5]
    )

    # Make sure we have a facet function on the heart mesh
    assert heart_ffun.size() == heart_mesh.num_facets()

    # Should be equally many marked facets
    assert (heart_ffun.array() == 4).sum() == (ffun.array() == 4).sum()
    assert (heart_ffun.array() == 5).sum() == (ffun.array() == 5).sum()
