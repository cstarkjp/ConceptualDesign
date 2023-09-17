"""
Various type definitions, enumerations, etc.

---------------------------------------------------------------------
"""
# Library
import warnings

# import logging
from numpy.typing import NDArray
import numpy as np
from enum import Enum

import trimesh as tm
from trimesh.base import Trimesh
from trimesh.caching import TrackedArray as TrimeshTrackedArray
import pyvista as pv
from pyvista import PolyData as PVMesh
from pyvista import Plotter as PVPlotter
from stl.mesh import Mesh as STLMesh
from networkx import Graph as NXGraph

warnings.filterwarnings("ignore")


