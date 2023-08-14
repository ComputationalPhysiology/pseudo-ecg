from importlib.metadata import metadata

from . import ecg
from . import eikonal
from . import mesh_utils

from .mesh_utils import (
    surface_to_volume_ratio,
    vertex_map_kdtree,
    vertex_map_meshview,
    vertex_map_submesh,
)

meta = metadata("pseudo-ecg")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = [
    "ecg",
    "eikonal",
    "mesh_utils",
    "surface_to_volume_ratio",
    "vertex_map_kdtree",
    "vertex_map_meshview",
    "vertex_map_submesh",
]
