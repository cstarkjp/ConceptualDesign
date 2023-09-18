"""
Class to build a model geometry.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`trimesh`
  -  :mod:`networkx`

---------------------------------------------------------------------
"""
# Library
import warnings
import logging
from typing import (
    Dict, Any, Tuple, Optional, List, Callable, Iterable, Sized, Generator,
)

import os
import numpy as np
import trimesh

from cdsn.definitions import (
    NDArray, Trimesh, TrimeshTrackedArray, NXGraph
)

warnings.filterwarnings("ignore")

__all__ = ["Mesh"]

class Mesh:
    """
    Class to build a mesh geometry from an STL file.

    Args:
        name (str):
            name of source geometry STL file (stem only)
        data_path (optional str):
            relative path from here to data STL files (assumed to be ../Data/STL/)

    Attributes:
        file_path_name (str): 
            relative path to STL file and its name with ".stl" extension
        trimesh (Trimesh):
            triangular mesh imported from STL file in Trimesh format
    """
    def __init__(
            self,
            name: str,
            data_path: str = os.path.join(os.pardir,"Data","STL",),
        ) -> None:
        # Read model from STL file
        self.name = name
        self.read_from_stl(data_path, name, )

    def read_from_stl(
            self,
            data_path: str,
            name: str,
        ) -> None:
        """
        Use Trimesh to load an STL file.

        Args:
            data_path (str):
                relative path to the data folder containing the STL file
            name (str):
                STL file name

        Attributes:
            file_path_name (str):
                relative path and name of STL file
            trimesh (Trimesh):
                processed mesh as a Trimesh object
        """
        self.file_path_name: str = os.path.join(data_path,f"{name}.stl")
        self.trimesh: Trimesh = trimesh.load(self.file_path_name, process=True,)
