"""
Class to partition the communities derived from a mesh-based graph into ground, members and applied loads.

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

import networkx as nx

from cdsn.definitions import (
    NDArray, Trimesh, TrimeshTrackedArray, NXGraph
)
from cdsn.communities import Communities

warnings.filterwarnings("ignore")

__all__ = ["Topology"]

class Topology:
    """
    Class to partition the communities derived from a mesh-based graph into ground, members and applied loads.

    Args:
        communities (Community):
            name of source topology STL file (stem only)

    Attributes:
        groundcommunity (int):
            community-index of ground community
        groundcommunity_vertices (frozenset[int]):
            immutable set of all vertices in the ground community
        groundcommunity_triangles (frozenset[int]):
            immutable set of all triangles in the ground community
        groundcommunity_area (float):
            total area of the ground community mesh
        appliedload_communities (frozenset[int]):
            immutable set of communities that are actually applied loads
        d_appliedload_communities (dict):
            applied-force-indexed dictionary of communities that are actually applied loads
        d_appliedload_trivertices (dict):
            applied-force-indexed dictionary of triangles representing applied loads
        d_member_vertices (dict):
            member-indexed dictionary of vertices in each member
        d_member_community (dict):
            dictionary listing members with communities
        d_community_member (dict):
            dictionary listing communities with members (reverse look-up)
        nodes (frozenset):
            immutable set of all nodes (hub vertices linking distinct members) 
        d_node_communities (dict):
            dictionary listing nodes (hub vertices linking distinct members) with the connected communities
        d_node_members (dict):
            dictionary listing nodes (hub vertices linking distinct members) with the connected members
        d_member_nodes (dict):
            dictionary listing members with all their nodes
        d_appliedload_node (dict[int,int]):
            dictionary listing applied loads with their connected nodes
        d_node_appliedload (dict[int,int]):
            dictionary listing nodes with their applied loads (reverse look-up)
        d_member_appliedloads (dict):
            dictionary listing members with all their applied loads
        d_member_appliedloadnodes (dict):
            dictionary listing members with all their applied load nodes
    """
    def __init__(
            self,
            communities: Communities,
        ) -> None:
        self.communities = communities 
        self.find_groundcommunity() 
        self.split_into_ground_appliedloads_members()
        self.find_nodes_communities()
        self.find_nodes_members()
        self.find_members_nodes()
        self.find_appliedloads_nodes()
        self.find_members_appliedloads()

    def find_groundcommunity(self) -> None:
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
            = max(self.communities.d_community_areas, key=self.communities.d_community_areas.__getitem__)

    def split_into_ground_appliedloads_members(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.groundcommunity_vertices: frozenset[int] \
            = self.communities.d_community_vertices[self.groundcommunity]
        self.groundcommunity_triangles: frozenset[int] = \
            self.communities.d_community_triangles[self.groundcommunity]
        self.groundcommunity_area: float = \
            self.communities.d_community_areas[self.groundcommunity]
        self.appliedload_communities: frozenset[int] = frozenset([
            community_
            for community_,vertices_ in self.communities.d_community_vertices.items()
            if len(vertices_)==3
        ])
        self.d_appliedload_communities: Dict = {
            appliedload_: community_ 
            for appliedload_,community_ in enumerate(self.appliedload_communities)
        }
        self.d_appliedload_trivertices: Dict = {
            appliedload_: self.communities.d_community_vertices[community_] 
            for appliedload_,community_ in self.d_appliedload_communities.items()
        }
        d_community_vertices_: Dict = self.communities.d_community_vertices.copy()
        del d_community_vertices_[self.groundcommunity]
        for community_ in self.appliedload_communities:
            del d_community_vertices_[community_]
        self.d_member_vertices: Dict = {
            member_: vertices_ 
            for member_,(_,vertices_) in enumerate(d_community_vertices_.items())
        }
        self.d_member_community: Dict = {
            member_: community_ 
            for member_,(community_,_) in enumerate(d_community_vertices_.items())
        }
        self.d_community_member: Dict = {
            community_: member_  for member_,community_ in self.d_member_community.items()
        }

    def find_nodes_communities(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        d: Dict =  dict(self.build_nodes_dict())
        self.d_node_communities: Dict \
            = dict(sorted(d.items(), key=lambda item: item[0]))
        self.nodes = frozenset(self.d_node_communities)

    def build_nodes_dict(self) -> Iterable[Tuple[int, frozenset]]:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        community_vertices = self.communities.d_community_vertices
        for target_community_, target_vertices_ in community_vertices.items():
            othercommunity_vertices: Dict = community_vertices.copy()
            del othercommunity_vertices[target_community_]
            # print((target_community_, target_vertices_), list(othercommunity_vertices.items()), )
            # Now we have (1) a target community (2) the other communities
            # Search through all the vertices of the target community
            for target_vertex_ in target_vertices_:
                # Use set intersection to check if the target community
                #   and the current other community share this vertex => node
                connected_communities_: List = [target_community_] + [
                    othercommunity_
                    for othercommunity_, othervertices_ in othercommunity_vertices.items()
                    if len(frozenset((target_vertex_,)).intersection(othervertices_))>0
                ]
                # print(f"Connected? {connected_communities_}")
                if len(connected_communities_)>1:
                    yield(target_vertex_, frozenset(sorted(connected_communities_)))

    def find_nodes_members(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_node_members: Dict = {
            node_: frozenset([
                self.d_community_member[community_]
                if community_ in self.d_community_member else (
                    "force" if community_ in self.d_appliedload_communities.values()
                    else "ground"
                )
                for community_ in communities_
            ])
            for node_, communities_ in self.d_node_communities.items()
        }

    def find_members_nodes(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_member_nodes: Dict = {
            member_: frozenset(
                vertices_.intersection(self.nodes)
            )
            for member_, vertices_ in self.d_member_vertices.items()
        }

    def find_appliedloads_nodes(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_appliedload_node: Dict = {
            appliedload_: [
                node_
                for node_, connected_communities_ in self.d_node_communities.items()
                if community_ in connected_communities_
            ][0]
            for appliedload_, community_ in self.d_appliedload_communities.items()
        }
        self.d_node_appliedload: Dict = {
            node_: appliedload_
            for appliedload_,node_ in self.d_appliedload_node.items()
        }

    def find_members_appliedloads(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_member_appliedloadnodes: Dict = {
            member_: nodes_.intersection(frozenset([
                appliedloadnode_ 
                for appliedload_,appliedloadnode_ in self.d_appliedload_node.items()
            ]))
            for member_,nodes_ in self.d_member_nodes.items()
        }
        self.d_member_appliedloads: Dict = {
            member_: frozenset([
                self.d_node_appliedload[appliedloadnode_]
                for appliedloadnode_ in appliedloadnodes_
            ])
            for member_,appliedloadnodes_ in self.d_member_appliedloadnodes.items()
        }
