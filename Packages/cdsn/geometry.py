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

__all__ = ["Geometry"]

class Geometry:
    """
    Class to partition the communities derived from a mesh-based graph into ground, members and applied loads.

    Args:
        communities (Community):
            name of source geometry STL file (stem only)

    Attributes:
        groundcommunity (int):
            community-index of ground community
        groundcommunity_nodes (frozenset[int]):
            immutable set of all nodes in the ground community
        groundcommunity_triangles (frozenset[int]):
            immutable set of all triangles in the ground community
        groundcommunity_area (float):
            total area of the ground community mesh
        appliedload_communities (frozenset[int]):
            immutable set of communities that are actually applied loads
        d_appliedload_communities (dict):
            applied-force-indexed dictionary of communities that are actually applied loads
        d_appliedload_trinodes (dict):
            applied-force-indexed dictionary of triangles representing applied loads
        d_member_nodes (dict):
            member-indexed dictionary of nodes in each member
        d_member_community (dict):
            dictionary listing members with communities
        d_community_member (dict):
            dictionary listing communities with members (reverse look-up)
        keynodes (frozenset):
            immutable set of all keynodes (hub nodes linking distinct members) 
        d_keynode_communities (dict):
            dictionary listing keynodes (hub nodes linking distinct members) with the connected communities
        d_keynode_members (dict):
            dictionary listing keynodes (hub nodes linking distinct members) with the connected members
        d_member_keynodes (dict):
            dictionary listing members with all their keynodes
        d_appliedload_keynode (dict[int,int]):
            dictionary listing applied loads with their connected keynodes
        d_keynode_appliedload (dict[int,int]):
            dictionary listing keynodes with their applied loads (reverse look-up)
        d_member_appliedloads (dict):
            dictionary listing members with all their applied loads
        d_member_appliedloadkeynodes (dict):
            dictionary listing members with all their applied load keynodes
    """
    def __init__(
            self,
            communities: Communities,
        ) -> None:
        self.cmnts = communities 
        self.find_groundcommunity() 
        self.split_into_ground_appliedloads_members()
        self.find_keynodes_communities()
        self.find_keynodes_members()
        self.find_members_keynodes()
        self.find_appliedloads_keynodes()
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
            = max(self.cmnts.d_community_areas, key=self.cmnts.d_community_areas.__getitem__)

    def split_into_ground_appliedloads_members(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.groundcommunity_nodes: frozenset[int] \
            = self.cmnts.d_community_nodes[self.groundcommunity]
        self.groundcommunity_triangles: frozenset[int] = \
            self.cmnts.d_community_triangles[self.groundcommunity]
        self.groundcommunity_area: float = \
            self.cmnts.d_community_areas[self.groundcommunity]
        self.appliedload_communities: frozenset[int] = frozenset([
            community_
            for community_,nodes_ in self.cmnts.d_community_nodes.items()
            if len(nodes_)==3
        ])
        self.d_appliedload_communities: Dict = {
            appliedload_: community_ 
            for appliedload_,community_ in enumerate(self.appliedload_communities)
        }
        self.d_appliedload_trinodes: Dict = {
            appliedload_: self.cmnts.d_community_nodes[community_] 
            for appliedload_,community_ in self.d_appliedload_communities.items()
        }
        d_community_nodes_: Dict = self.cmnts.d_community_nodes.copy()
        del d_community_nodes_[self.groundcommunity]
        for community_ in self.appliedload_communities:
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

    def find_keynodes_communities(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        d: Dict =  dict(self.build_keynodes_dict())
        self.d_keynode_communities: Dict \
            = dict(sorted(d.items(), key=lambda item: item[0]))
        self.keynodes = frozenset(self.d_keynode_communities)

    def find_keynodes_members(self) -> None:
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
                    "force" if community_ in self.d_appliedload_communities.values()
                    else "ground"
                )
                for community_ in communities_
            ])
            for keynode_, communities_ in self.d_keynode_communities.items()
        }

    def find_members_keynodes(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_member_keynodes: Dict = {
            member_: frozenset(
                nodes_.intersection(self.keynodes)
            )
            for member_, nodes_ in self.d_member_nodes.items()
        }

    def build_keynodes_dict(self) -> Iterable[Tuple[int, frozenset]]:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        for community_, nodes_ in self.cmnts.d_community_nodes.items():
            d_othercommunity_nodes: Dict = self.cmnts.d_community_nodes.copy()
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

    def find_appliedloads_keynodes(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_appliedload_keynode: Dict = {
            appliedload_: [
                keynode_
                for keynode_, connected_communities_ in self.d_keynode_communities.items()
                if community_ in connected_communities_
            ][0]
            for appliedload_, community_ in self.d_appliedload_communities.items()
        }
        self.d_keynode_appliedload: Dict = {
            keynode_: appliedload_
            for appliedload_,keynode_ in self.d_appliedload_keynode.items()
        }

    def find_members_appliedloads(self) -> None:
        """
        XXX

        Attributes:
            XXX (XXX):
                XXX
        """
        self.d_member_appliedloadkeynodes: Dict = {
            member_: keynodes_.intersection(frozenset([
                appliedloadkeynode_ 
                for appliedload_,appliedloadkeynode_ in self.d_appliedload_keynode.items()
            ]))
            for member_,keynodes_ in self.d_member_keynodes.items()
        }
        self.d_member_appliedloads: Dict = {
            member_: frozenset([
                self.d_keynode_appliedload[appliedloadkeynode_]
                for appliedloadkeynode_ in appliedloadkeynodes_
            ])
            for member_,appliedloadkeynodes_ in self.d_member_appliedloadkeynodes.items()
        }
