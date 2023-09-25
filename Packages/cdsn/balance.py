"""
Class to compute force & torque balance.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`sympy`

---------------------------------------------------------------------
"""
# Library
import warnings
import logging
from typing import (
    Dict, Any, Tuple, Optional, List, Callable, Iterable, Sized, Generator,
)

import numpy as np
from sympy import (
    Eq, Symbol, MatrixSymbol, MatAdd, Matrix, Rational, Integer,
)

from cdsn.definitions import (
    NDArray, Trimesh, TrimeshTrackedArray, NXGraph
)
from cdsn.forces import Forces

warnings.filterwarnings("ignore")

__all__ = ["Balance"]

class Balance:
    """
    Class to compute force & torque balance.

    Args:
        forces (Force):
            forces/torques/loads object

    Attributes:
        XXX (XXX):
            XXX
    """
    def __init__(
            self,
            forces: Forces,
        ) -> None:
        self.forces = forces 
        self.setup_node_fvc()
        self.setup_member_fvc()
        self.compute_member_forceinfo()
        self.compute_member_forcemoment()

    def setup_node_fvc(self) -> None:
        topology = self.forces.topology
        self.d_node_fvc = {
            node_: {
                "X": Symbol(rf"F_{node_},X", real=True,),
                "Y": Symbol(rf"F_{node_},Y", real=True,),
            }
            for node_ in topology.d_node_communities
        }

    def setup_member_fvc(self) -> None:
        topology = self.forces.topology
        self.d_member_fvc = {
            member_: [self.d_node_fvc[node_] for node_ in nodes_]
            for member_, nodes_ in topology.d_member_nodes.items()
        }

    def compute_member_forceinfo(self) -> None:
        n_round = 6
        topology = self.forces.topology
        graph = topology.communities.graph
        self.d_member_forceinfo = {
            member_: (
                M := MatrixSymbol(r"\mathbf{M}"+f"_{member_}",3,len(fvc_),),
                M_ := Matrix([
                    [1]*len(fvc_)+[0]*len(fvc_),
                    [0]*len(fvc_)+[1]*len(fvc_),
                    [np.round(graph.d_vertex_vpoints[node_][0],n_round) for node_ in nodes_]
                    +[np.round(graph.d_vertex_vpoints[node_][1],n_round) for node_ in nodes_]
                ]),
                F := MatrixSymbol(r"\mathbf{F}"+f"_{member_}",len(fvc_),1),
                F_ := Matrix(
                    [
                        fvxy_["X"]
                        for fvxy_ in fvc_
                    ] + [
                        fvxy_["Y"]
                        for fvxy_ in fvc_
                    ]
                )
            )
            for ((member_,fvc_),(member_,nodes_))
            in zip(self.d_member_fvc.items(),topology.d_member_nodes.items())
        }

    def compute_member_forcemoment(self) -> None:
        self.d_member_forcemoment = {
            member_: Eq(
                forceinfo_[0]*forceinfo_[2],
                forceinfo_[1]*forceinfo_[3]
            )
            for member_, forceinfo_ in self.d_member_forceinfo.items()
        }
