from importlib.metadata import metadata

from . import ecg
from . import eikonal
from . import mesh_utils

meta = metadata("pseudo-ecg")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = ["ecg", "eikonal", "mesh_utils"]
