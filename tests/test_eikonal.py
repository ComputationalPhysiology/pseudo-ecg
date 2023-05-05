import dolfin
import numpy as np

from pseudo_ecg import eikonal


def test_eikonal_2D():
    mesh = dolfin.UnitSquareMesh(20, 20)
    electrode = np.array([0.0, 0.0])

    dist = eikonal.distance(mesh, electrode)

    rtol = 0.01
    assert np.isclose(dist(1.0, 1.0), np.sqrt(2), rtol=rtol)
    assert np.isclose(dist(1.0, 0.0), 1.0, rtol=rtol)
    assert np.isclose(dist(0.0, 1.0), 1.0, rtol=rtol)
    assert np.isclose(dist(0.0, 0.0), 0.0, rtol=rtol)


def test_eikonal_2D_rectangle():
    mesh = dolfin.RectangleMesh(
        dolfin.Point(-1.0, -1.0), dolfin.Point(1.0, 1.0), 100, 100
    )
    electrode = np.array([1.0, 1.0])

    dist = eikonal.distance(mesh, electrode, factor=15.0)

    rtol = 0.01
    assert np.isclose(dist(0.0, 0.0), np.sqrt(2), rtol=rtol)
    assert np.isclose(dist(1.0, 0.0), 1.0, rtol=rtol)
    assert np.isclose(dist(0.0, 1.0), 1.0, rtol=rtol)
    assert np.isclose(dist(1.0, 1.0), 0.0, rtol=rtol)


def test_eikonal_3D():
    mesh = dolfin.UnitCubeMesh(5, 5, 5)
    electrode = np.array([0.0, 0.0, 0.0])

    dist = eikonal.distance(mesh, electrode)

    rtol = 0.05
    assert np.isclose(dist(1.0, 1.0, 1.0), np.sqrt(3), rtol=rtol)
    assert np.isclose(dist(1.0, 0.0, 0.0), 1.0, rtol=rtol)
    assert np.isclose(dist(0.0, 1.0, 0.0), 1.0, rtol=rtol)
    assert np.isclose(dist(0.0, 0.0, 0.0), 0.0, rtol=rtol)


if __name__ == "__main__":
    test_eikonal_3D()
