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
        self.compute_appliedloads_forcevectors()

    def compute_force_vector(
            self,
            node: int, 
            trinodes: frozenset,
        ) -> NDArray:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        graph = self.geometry.cmnts.graph
        node_vertex = np.array(graph.d_node_vertices[node])
        force_vectors = [
            node_vertex-np.array(graph.d_node_vertices[trinode_])
            for trinode_ in trinodes
            if trinode_!=node
        ]
        net_force_vector = sum(force_vectors)
        unit_net_force_vector = net_force_vector/np.linalg.norm(net_force_vector)
        return np.round(unit_net_force_vector,6)
    
    def compute_appliedloads_forcevectors(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_appliedload_forcevector: Dict = {
            appliedload_: self.compute_force_vector(
                node=self.geometry.d_appliedload_keynode[appliedload_],
                trinodes=trinodes_
            )
            for appliedload_, trinodes_ in self.geometry.d_appliedload_trinodes.items()
        }