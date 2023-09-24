"""
Class to classify applied loads, forces and torques at nodes.

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
from cdsn.topology import Topology

warnings.filterwarnings("ignore")

__all__ = ["Forces"]

class Forces:
    """
    Class to classify applied loads, forces and torques at nodes.

    Args:
        topology (Topology):
            model topology object

    Attributes:
        XXX (XXX):
            XXX
    """
    def __init__(
            self,
            topology: Topology,
        ) -> None:
        self.topology = topology 
        self.compute_appliedloads_forcevectors()
        self.find_appliedloads_triangles()

    def compute_force_vector(
            self,
            vertex: int, 
            trivertices: frozenset,
        ) -> NDArray:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        graph = self.topology.communities.graph
        vertex_vertex = np.array(graph.d_vertex_vpoints[vertex])
        force_vectors = [
            vertex_vertex-np.array(graph.d_vertex_vpoints[trivertex_])
            for trivertex_ in trivertices
            if trivertex_!=vertex
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
                vertex=self.topology.d_appliedload_node[appliedload_],
                trivertices=trivertices_
            )
            for appliedload_, trivertices_ in self.topology.d_appliedload_trivertices.items()
        }

    def find_appliedloads_triangles(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        graph = self.topology.communities.graph
        self.d_appliedload_triangle: Dict = {
            appliedload_: graph.d_trivertices_triangle[trivertices_]
            for appliedload_,trivertices_ in self.topology.d_appliedload_trivertices.items()
        }
        self.d_triangle_appliedload: Dict = {
            triangle_: appliedload_
            for appliedload_,triangle_ in self.d_appliedload_triangle.items()
        }