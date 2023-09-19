"""
Class to compute forces and torques at keynodes.

---------------------------------------------------------------------

Requires Python packages:

---------------------------------------------------------------------
"""
# Library
import warnings
import logging
from typing import (
    Dict, Any, Tuple, Optional, List, Callable, Iterable, Sized, Generator,
)

import numpy as np

from cdsn.definitions import (
    NDArray, Trimesh, TrimeshTrackedArray, NXGraph
)
from cdsn.geometry import Geometry

warnings.filterwarnings("ignore")

__all__ = ["Forces"]

class Forces:
    """
    Class to compute forces and torques at keynodes.

    Args:
        geometry (Geometry):
            model geometry object

    Attributes:
        XXX (XXX):
            XXX
    """
    def __init__(
            self,
            geometry: Geometry,
        ) -> None:
        self.geometry = geometry 
