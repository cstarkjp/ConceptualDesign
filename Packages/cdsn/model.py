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
    """
    Class to build mesh geometry and analyze it.

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
        edges (NDArray[int,int]):
            array of pairs of node indexes indicating edge connections
        edge_lengths (TrimeshTrackedArray[float]):
            lengths of those node-node connections
        graph (NXGraph):
            mesh converted into a networkx graph
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
        d_community_nodes (dict):
            dictionary of nodes grouped and indexed by their 3-clique community
        n_communities (int):
            number of 3-clique communities
        d_community_triangles (dict):
            community-indexed dictionary of constituent triangles
        d_community_areas (dict):
            community-indexed dictionary of total community areas
        groundcommunity (int):
            community-index of ground community
        triangle_areas (NDArray):
            areas of all the mesh triangles
        groundcommunity_nodes (frozenset[int]):
            immutable set of all nodes in the ground community
        groundcommunity_triangles (frozenset[int]):
            immutable set of all triangles in the ground community
        groundcommunity_area (float):
            total area of the ground community mesh
        appliedforce_communities (frozenset[int]):
            immutable set of communities that are actually applied forces
        d_appliedforce_communities (dict):
            applied-force-indexed dictionary of communities that are actually applied forces
        d_appliedforce_trinodes (dict):
            applied-force-indexed dictionary of triangles representing applied forces
        d_member_nodes (dict):
            member-indexed dictionary of nodes in each member
        d_member_community (dict):
            dictionary linking members to communities
        d_community_member (dict):
            dictionary linking communities to members (reverse look-up)
        d_keynode_communities (dict):
            dictionary linking keynodes (hub nodes linking distinct members) to the connected communities
        d_keynode_members (dict):
            dictionary linking keynodes (hub nodes linking distinct members) to the connected members
        d_appliedforce_keynode (dict[int,frozenset[int/str]]):
            dictionary linking applied forces to their connected keynodes
    """
    def __init__(
            self,
            name: str,
            data_path: str = os.path.join(os.pardir,"Data","STL",),
        ):
        # Read model from STL file
        self.name = name
        self.read_from_stl(data_path, name, )

        # Use networkx to build a graph from the model edges
        self.build_graph()

        # Then analyze the graph to find all the triangles and how they cluster
        self.tolerance: float = 1e-10
        self.find_vertices()
        self.find_triangles()
        self.compute_triangle_areas()
        self.find_community_nodes()
        self.find_community_triangles()
        self.find_community_areas()
        self.find_groundcommunity()
        self.split_into_ground_appliedforces_members()
        self.find_keynodes_communities()
        self.find_keynodes_members()
        self.find_keynodes_for_appliedforces()

    def chop(self, array: NDArray):
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

    def read_from_stl(
            self,
            data_path: str,
            name: str,
        ):
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

    def build_graph(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.edges: NDArray = self.trimesh.edges_unique
        self.edge_lengths: TrimeshTrackedArray[float] = self.trimesh.edges_unique_length
        self.graph: NXGraph = nx.Graph()
        for edge_, length_ in zip(self.edges, self.edge_lengths):
            self.graph.add_edge(*edge_, length=length_)
        
    def find_vertices(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.vertices: TrimeshTrackedArray[float,float,float] = self.trimesh.vertices
        self.d_node_vertices: Dict[int,NDArray] = {
            key_: np.array(vertices_)
            for key_,vertices_ in enumerate(self.trimesh.vertices)
        }

    def find_triangles(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        triangles_: Generator = nx.simple_cycles(self.graph, length_bound=3,)
        self.d_triangle_trinodes: Dict = {
            key_: tuple(sorted(triangle_)) 
            for key_,triangle_ in enumerate(list(triangles_))
        }
        self.d_trinodes_triangles: Dict = {
            nodes_: key_ for key_, nodes_ in self.d_triangle_trinodes.items()
        }
        self.n_triangles: int = max(self.d_triangle_trinodes)+1
    
    def find_community_nodes(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_community_nodes: Dict = {
            key_: community_ 
            for key_, community_ in enumerate(
                nx.community.k_clique_communities(self.graph,3)
            )
        }
        self.n_communities: int = max(self.d_community_nodes)+1

    def find_community_triangles(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_community_triangles: Dict = {
            key_: frozenset(list(
                self.find_triangles_in(self.d_triangle_trinodes, community_,)
            ))
            for key_,community_ in self.d_community_nodes.items()
        }

    def find_community_areas(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_community_areas: Dict = {
            key_: np.sum(np.array([
                area(self.chop(self.vertices[np.r_[triangle_]]))            
                for triangle_ in triangles_
            ]))
            for key_,triangles_ in self.d_community_triangles.items()
        }

    def find_groundcommunity(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        # Using .__getitem__ instead of .get to make mypy happy:
        # https://stackoverflow.com/questions/75365839/mypy-with-dictionarys-get-function
        self.groundcommunity: int \
            = max(self.d_community_areas, key=self.d_community_areas.__getitem__)

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
        triangle_nodes_: Tuple[int,int,int]
        for triangle_nodes_ in triangles.values():
            n_shared_nodes = sum([
                is_node_in(triangle_node_, community)
                for triangle_node_ in triangle_nodes_
            ])
            if n_shared_nodes==3:
                yield(triangle_nodes_)

    def compute_triangle_areas(self):
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

    def split_into_ground_appliedforces_members(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.groundcommunity_nodes: frozenset[int] \
            = self.d_community_nodes[self.groundcommunity]
        self.groundcommunity_triangles: frozenset[int] = \
            self.d_community_triangles[self.groundcommunity]
        self.groundcommunity_area: float = \
            self.d_community_areas[self.groundcommunity]
        self.appliedforce_communities: frozenset[int] = frozenset([
            community_
            for community_,nodes_ in self.d_community_nodes.items()
            if len(nodes_)==3
        ])
        self.d_appliedforce_communities: Dict = {
            appliedforce_: community_ 
            for appliedforce_,community_ in enumerate(self.appliedforce_communities)
        }
        self.d_appliedforce_trinodes: Dict = {
            appliedforce_: self.d_community_nodes[community_] 
            for appliedforce_,community_ in self.d_appliedforce_communities.items()
        }
        d_community_nodes_: Dict = self.d_community_nodes.copy()
        del d_community_nodes_[self.groundcommunity]
        for community_ in self.appliedforce_communities:
            del d_community_nodes_[community_]
        self.d_member_nodes: Dict = {
            member_: nodes_ 
            for member_,(_,nodes_) in enumerate(d_community_nodes_.items())
        }
        self.d_member_community: Dict = {
            member_: community_ 
            for member_,(community_,_) in enumerate(d_community_nodes_.items())
        }
        self.d_community_member: Dict = {
            community_: member_  for member_,community_ in self.d_member_community.items()
        }

    def find_keynodes_communities(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        d: Dict =  dict(self.build_keynodes_dict())
        self.d_keynode_communities: Dict \
            = dict(sorted(d.items(), key=lambda item: item[0]))

    def find_keynodes_members(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_keynode_members: Dict = {
            keynode_: frozenset([
                self.d_community_member[community_]
                if community_ in self.d_community_member else (
                    "force" if community_ in self.d_appliedforce_communities.values()
                    else "ground"
                )
                for community_ in communities_
            ])
            for keynode_, communities_ in self.d_keynode_communities.items()
        }

    def build_keynodes_dict(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        for community_, nodes_ in self.d_community_nodes.items():
            d_othercommunity_nodes: Dict = self.d_community_nodes.copy()
            del d_othercommunity_nodes[community_]
            # Now we have (1) a target community (2) the other communities
            # Search through all the nodes of the target community
            for node_ in nodes_:
                # Use set intersection to check if the target community
                #   and the current other community share this node => keynode
                connected_communities_: List = [community_] + [
                    othercommunity_
                    for othercommunity_, othernodes_ in d_othercommunity_nodes.items()
                    if len(frozenset((node_,)).intersection(othernodes_))>0
                ]
                if len(connected_communities_)>1:
                    yield(node_, frozenset(sorted(connected_communities_)))

    def find_keynodes_for_appliedforces(self):
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_appliedforce_keynode: Dict = {
            appliedforce_: [
                keynode_
                for keynode_, connected_communities_ in self.d_keynode_communities.items()
                if community_ in connected_communities_
            ][0]
            for appliedforce_, community_ in self.d_appliedforce_communities.items()
        }
