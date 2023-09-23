"""
Class to partition a mesh-based graph into 3-clique communities.

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
from cdsn.graph import Graph

warnings.filterwarnings("ignore")

__all__ = ["Communities"]

is_vertex_in = lambda triangle_,community_: 1 if triangle_ in community_ else 0
area = lambda v: np.abs(np.dot((v[1]-v[0]),(v[1]-v[0])))

class Communities:
    """
    Class to partition a mesh-based graph into 3-clique communities.

    Args:
        graph (Graph):
            object containing nxgraph built from mesh and related properties

    Attributes:
        d_community_vertices (dict):
            dictionary of vertices grouped and indexed by their 3-clique community
        n_communities (int):
            number of 3-clique communities
        d_community_triangles (dict):
            community-indexed dictionary of constituent triangles
        d_community_areas (dict):
            community-indexed dictionary of total community areas
        triangle_areas (NDArray):
            areas of all the mesh triangles
    """
    def __init__(
            self,
            graph: Graph,
        ) -> None:
        self.graph = graph
        self.find_community_vertices()
        self.find_community_triangles()
        self.find_community_areas()

    def find_community_vertices(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_community_vertices: Dict = {
            key_: community_ 
            for key_, community_ in enumerate(
                nx.community.k_clique_communities(self.graph.nxgraph,3)
            )
        }
        self.n_communities: int = max(self.d_community_vertices)+1

    def find_community_triangles(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_community_triangles: Dict = {
            key_: frozenset(list(
                self.find_triangles_in(self.graph.d_triangle_trivertices, community_,)
            ))
            for key_,community_ in self.d_community_vertices.items()
        }

    def find_community_areas(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_community_areas: Dict = {
            key_: np.sum(np.array([
                area(self.graph.chop(self.graph.vpoints[np.r_[tuple(triangle_)]]))            
                for triangle_ in triangles_
            ]))
            for key_,triangles_ in self.d_community_triangles.items()
        }

    @staticmethod
    def find_triangles_in(
            triangles: Dict, 
            community: frozenset,
        ) -> Generator:
        """
        XXX

        Args:
            XXX(XXX):
                XXX

        Returns:
            XXX (XXX):
                XXX
        """
        triangle_vertices_: Tuple[int,int,int]
        for triangle_vertices_ in triangles.values():
            n_shared_vertices = sum([
                is_vertex_in(triangle_vertex_, community)
                for triangle_vertex_ in triangle_vertices_
            ])
            if n_shared_vertices==3:
                yield(triangle_vertices_)