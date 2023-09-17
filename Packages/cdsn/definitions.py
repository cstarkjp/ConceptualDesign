"""
Various type definitions, enumerations, etc.

---------------------------------------------------------------------
"""
# Library
import warnings

# import logging
from typing import List, Tuple, Dict, Optional, Callable, TypeAlias, Union, TypeAlias
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

__all__ = ["LineIdx"]

# TriMesh: TypeAlias = tm.TrackedArray
# PVMesh: TypeAlias = NDArray

