"""
Class to convert mesh geometry into a networkx graph.

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
    Class to convert mesh geometry into a networkx graph.

    Args:
        mesh (Mesh):

    Attributes:
        graph (NXGraph):
            networkx graph converted from mesh read from STL file
        vertices (TrimeshTrackedArray[float,float,float]):
            x,y,z positions of graph nodes
        d_node_vertices (dict[int,NDArray[float,float,float]]):
            node-indexed dictionary of x,y,z vertices
        d_triangle_trinodes (dict[int,tuple[int,int,int]]):
            triangle-indexed dictionary of triangle nodes
        d_trinodes_triangles (dict[int,NDArray[float,float,float]]):
            triangle-node-indexed dictionary of triangles (for reverse look-up)
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
        self.find_vertices()
        self.find_triangles()
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
        self.edge_lengths: TrimeshTrackedArray[float] = self.mesh.trimesh.edges_unique_length
        self.nxgraph: NXGraph = nx.Graph()
        for edge_, length_ in zip(self.edges, self.edge_lengths):
            self.nxgraph.add_edge(*edge_, length=length_)
        
    def find_vertices(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.vertices: TrimeshTrackedArray[float,float,float] = self.mesh.trimesh.vertices
        self.d_node_vertices: Dict[int,NDArray] = {
            key_: np.array(vertices_)
            for key_,vertices_ in enumerate(self.vertices)
        }

    def find_triangles(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        triangles_: Generator = nx.simple_cycles(self.nxgraph, length_bound=3,)
        self.d_triangle_trinodes: Dict = {
            key_: tuple(sorted(triangle_)) 
            for key_,triangle_ in enumerate(list(triangles_))
        }
        self.d_trinodes_triangles: Dict = {
            nodes_: key_ for key_, nodes_ in self.d_triangle_trinodes.items()
        }
        self.n_triangles: int = max(self.d_triangle_trinodes)+1
    
    def compute_triangle_areas(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.triangle_areas: NDArray = np.array([
            area(self.chop(self.vertices[np.r_[triangle_]]))            
            for triangle_ in self.d_triangle_trinodes.values()
        ])    