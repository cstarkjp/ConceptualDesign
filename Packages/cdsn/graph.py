"""
Class to convert mesh topology into a networkx graph.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`networkx`

---------------------------------------------------------------------
"""
# Library
import warnings
import logging
from typing import (
    Dict, Any, Tuple, Optional, List, Callable, Iterable, Sized, Generator,
)

import numpy as np
import networkx as nx

from cdsn.definitions import (
    NDArray, Trimesh, TrimeshTrackedArray, NXGraph
)
from cdsn.mesh import Mesh

warnings.filterwarnings("ignore")

__all__ = ["Graph"]

area = lambda v: np.abs(np.dot((v[1]-v[0]),(v[1]-v[0])))

class Graph:
    """
    Class to convert mesh topology into a networkx graph.

    Args:
        mesh (Mesh):

    Attributes:
        graph (NXGraph):
            networkx graph converted from mesh read from STL file
        vpoints (TrimeshTrackedArray[float,float,float]):
            x,y,z positions of graph vertices
        d_vertex_vpoints (dict[int,NDArray[float,float,float]]):
            vertex-indexed dictionary of x,y,z vpoints
        d_vpoints_vertex:
            XXX
        d_triangle_trivertices (dict[int,frozenset[int,int,int]]):
            triangle-indexed dictionary of triangle vertices
        d_trivertices_triangle (dict[frozenset,int]):
            triangle-vertex-indexed dictionary of triangles (for reverse look-up)
        n_triangles (int):
            number of triangles in the mesh
    """
    def __init__(
            self,
            mesh: Mesh,
        ) -> None:
        self.mesh = mesh
        # Use networkx to build a graph from the model edges
        self.build_graph()
        # Then analyze the graph to find all the triangles and how they cluster
        self.tolerance: float = 1e-10
        self.round: int = 6
        self.label_vertices()
        self.label_triangles()
        self.compute_triangle_areas()

    def chop(self, array: NDArray) -> NDArray:
        """
        Chop tiny float values (close to tolerance) and set them to zero.

        Tolerance is set during instantiation of the class.

        Args:
            array (NDArray[float,...]):
                numpy array of whatever dimension

        Returns:
            array (NDArray[float,...]):
                chopped (cleaned) numpy array
        """
        chopped_array: NDArray = array.copy()
        chopped_array[np.isclose(array, 0, atol=self.tolerance)] = 0
        return chopped_array

    def build_graph(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.edges: NDArray = self.mesh.trimesh.edges_unique
        self.edge_lengths: TrimeshTrackedArray[float] \
            = self.mesh.trimesh.edges_unique_length
        self.nxgraph: NXGraph = nx.Graph()
        for edge_, length_ in zip(self.edges, self.edge_lengths):
            self.nxgraph.add_edge(*edge_, length=length_)
        
    def label_vertices(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.vpoints: TrimeshTrackedArray[float,float,float] \
            = np.round(self.mesh.trimesh.vertices, self.round)
        self.d_vertex_vpoints: Dict[int,tuple] = {
            vertex_: tuple(vpoints_)
            for vertex_,vpoints_ in enumerate(self.vpoints)
        }
        self.d_vpoints_vertex: Dict[int,tuple] = {
            vpoints_: vertex_
            for vertex_,vpoints_ in self.d_vertex_vpoints.items()
        }

    def label_triangles(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        trivertices_generator: Generator = nx.simple_cycles(self.nxgraph, length_bound=3,)
        self.d_triangle_trivertices: Dict = {
            triangle_: frozenset(sorted(trivertices_)) 
            for triangle_,trivertices_ in enumerate(list(trivertices_generator))
        }
        self.d_trivertices_triangle: Dict = {
            trivertices_: triangle_ 
            for triangle_, trivertices_ in self.d_triangle_trivertices.items()
        }
        self.n_triangles: int = max(self.d_triangle_trivertices)+1
    
    def compute_triangle_areas(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.triangle_areas: NDArray = np.array([
            area(self.chop(self.vpoints[np.r_[tuple(triangle_)]]))            
            for triangle_ in self.d_triangle_trivertices.values()
        ])    