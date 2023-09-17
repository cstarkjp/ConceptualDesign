"""
Class to build a model geometry.

---------------------------------------------------------------------

Requires Python packages/modules:
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
import networkx as nx

from cdsn.definitions import (
    NDArray, Trimesh, TrimeshTrackedArray, NXGraph
)

warnings.filterwarnings("ignore")

__all__ = ["Geometry"]

is_node_in = lambda triangle_,community_: 1 if triangle_ in community_ else 0
area = lambda v: np.abs(np.dot((v[1]-v[0]),(v[1]-v[0])))


class Geometry:
    def __init__(
            self,
            case_name: str,
            dist_max: float = 1e-3,
            data_path: str = os.path.join(os.pardir,"Data","STL",),
        ):
        # Read model from STL file
        self.name = case_name
        self.read_from_stl(data_path, case_name, )

        # Use networkx to build a graph from the model edges
        self.build_graph()

        # Then analyze the graph to find all the triangles and how they cluster
        self.tolerance: float = 1e-10
        self.find_vertices()
        self.find_triangles()
        self.compute_triangle_areas()
        self.find_communities()
        self.find_communities_triangles()
        self.find_communities_areas()
        self.find_ground_community()
        self.find_keynodes()

    def chop(self, array: NDArray):
        chopped_array: NDArray = array.copy()
        chopped_array[np.isclose(array, 0, atol=self.tolerance)] = 0
        return chopped_array

    def read_from_stl(
            self,
            data_path: str,
            case_name: str,
        ):
        self.file_name: str = os.path.join(data_path,f"{case_name}.stl")
        self.trimesh: Trimesh = trimesh.load(self.file_name, process=True,)

    def build_graph(self):
        self.edges: NDArray = self.trimesh.edges_unique
        self.edge_lengths: TrimeshTrackedArray = self.trimesh.edges_unique_length
        self.graph: NXGraph = nx.Graph()
        for edge_, length_ in zip(self.edges, self.edge_lengths):
            self.graph.add_edge(*edge_, length=length_)
        
    def find_vertices(self):
        self.vertices: TrimeshTrackedArray[float,float,float] = self.trimesh.vertices

    def find_triangles(self):
        triangles_: Generator = nx.simple_cycles(self.graph, length_bound=3,)
        self.triangles: Dict = {
            key_: tuple(sorted(triangle_)) 
            for key_,triangle_ in enumerate(list(triangles_))
        }
        self.triangles_by_nodes = {
            nodes_: key_ for key_, nodes_ in self.triangles.items()
        }
        self.n_triangles: int = max(self.triangles)+1
    
    def find_communities(self):
        self.communities: Dict = {
            key_: community_ 
            for key_, community_ in enumerate(
                nx.community.k_clique_communities(self.graph,3)
            )
        }
        self.n_communities: int = max(self.communities)+1

    def find_communities_triangles(self):
        self.communities_triangles: Dict = {
            key_: frozenset(list(
                self.find_triangles_in(self.triangles, community_,)
            ))
            for key_,community_ in self.communities.items()
        }

    def find_communities_areas(self):
        self.communities_areas: Dict = {
            key_: np.sum(np.array([
                area(self.chop(self.vertices[np.r_[triangle_]]))            
                for triangle_ in triangles_
            ]))
            for key_,triangles_ in self.communities_triangles.items()
        }

    def find_ground_community(self):
        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        self.ground_community: int \
            = max(self.communities_areas, key=self.communities_areas.get)

    @staticmethod
    def find_triangles_in(
            triangles: Dict, 
            community: frozenset,
        ) -> Generator:
        triangle_nodes_: Tuple[int,int,int]
        for triangle_nodes_ in triangles.values():
            n_shared_nodes = sum([
                is_node_in(triangle_node_, community)
                for triangle_node_ in triangle_nodes_
            ])
            if n_shared_nodes==3:
                yield(triangle_nodes_)

    def compute_triangle_areas(self):
        self.triangle_areas: NDArray = np.array([
            area(self.chop(self.vertices[np.r_[triangle_]]))            
            for triangle_ in self.triangles.values()
        ])    

    def find_keynodes(self):
        self.keynodes = dict(self.build_keynodes_dict())

    def build_keynodes_dict(self):
        for community_id_, nodes_ in self.communities.items():
            other_communities = self.communities.copy()
            del other_communities[community_id_]
            for other_community_id_, other_nodes_ in other_communities.items():
                for node_ in nodes_:
                    if node_ in other_nodes_:
                        yield(node_, (community_id_,other_community_id_))