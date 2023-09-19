"""
Provide a data visualization class.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`pyvista`
  -  :mod:`matplotlib`

---------------------------------------------------------------------
"""
# Library
import warnings
import logging
from itertools import cycle
import operator as op
from typing import Dict, Any, Tuple, Optional, List, Callable, Iterable, Sized
import numpy as np

import pyvista as pv
import networkx as nx

# MatPlotLib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.colors import to_rgb

from cdsn.definitions import (
    NDArray, Trimesh, TrimeshTrackedArray, STLMesh, NXGraph, 
    PVMesh, PVPlotter,
)
from cdsn.graph import Graph
from cdsn.communities import Communities
from cdsn.geometry import Geometry

warnings.filterwarnings("ignore")

__all__ = ["GraphingBase"]

color_cycle = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
color = lambda i: color_cycle[i % len(color_cycle)]
loop = lambda triangle: list(triangle) + [triangle[0]]

class GraphingBase:
    """
    Provide a visualization base class.

    Args:
        dpi (optional int):
            resolution for rasterized images
        font_size (optional int):
            general font size
        font_name (optional str):
            choice of font family

    Attributes:
        dpi (int):
            resolution for rasterized images
        font_size (int):
            general font size
        fdict  (dict):
            dictionary to which each figure is appended as it is generated
        colors  (list):
            list of colors
        n_colors  (int):
            number of colors
        color_cycle  (:obj:`itertools cycle <itertools.cycle>`):
            color property cycle
        markers  (list):
            list of markers
        n_markers  (:obj:`itertools cycle <itertools.cycle>`):
            number of markers
        marker_cycle  (int):
            cycle of markers
        linestyle_list  (list):
            list of line styles (solid, dashdot, dashed, custom dashed)
        color (:obj:`lambda(i) <lambda>`):
            return i^th color
        marker (:obj:`lambda(i) <lambda>`):
            return i^th marker
    """
    dpi: int
    font_size: int
    fdict: Dict
    colors: Callable
    n_colors: int
    color_cycle: Callable
    markers: Tuple
    n_markers: int
    marker_cycle: cycle
    linestyle_list: Tuple
    color: Callable
    marker: Callable
    font_family: str

    def __init__(
            self, 
            dpi: int = 100, 
            font_size: int = 11, 
            font_name: str="Arial",
    ) -> None:
        """Visualization base class."""
        self.dpi = dpi
        self.font_size = font_size
        self.fdict: Dict[Any, Any] = {}
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        self.colors = prop_cycle.by_key()["color"]  # type: ignore
        self.n_colors = len(self.colors)  # type: ignore
        self.color_cycle = cycle(self.colors)  # type: ignore
        self.markers = ("o", "s", "v", "p", "*", "D", "X", "^", "h", "P")
        self.n_markers = len(self.markers)
        self.marker_cycle = cycle(self.markers)
        self.linestyle_list = ("solid", "dashdot", "dashed", (0, (3, 1, 1, 1)))

        color_ = lambda i_: self.colors[i_ % self.n_colors]  # type: ignore
        marker_ = lambda i_: self.markers[i_ % self.n_markers]  # type: ignore
        self.color = color_  # type: ignore
        self.marker = marker_  # type: ignore
        # locale.setlocale(locale.LC_NUMERIC, "de_DE")
        # mpl.rc('text', usetex=True)
        self.font_family = font_name if font_name is not None else ("Arial" if "Arial" in self.get_fonts("Arial") else "")
        mpl.rc("font", size=self.font_size, family=self.font_family)

    def get_fonts(self, font: str) -> List[str]:
        """Fetch the names of all the font families available on the system."""
        fpaths = matplotlib.font_manager.findSystemFonts()
        fonts: List[str] = []
        for fpath in fpaths:
            if font is None or font in fpath:
                try:
                    # print(font, fpath)
                    font = matplotlib.font_manager.get_font(fpath).family_name
                    fonts.append(font)
                except RuntimeError as re:
                    logging.debug(f"{re}: failed to get font name for {fpath}")
                    pass
        return fonts

    def create_figure(
        self,
        fig_name: str,
        sub_plots = None,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
    ) -> plt.Figure:
        """
        Initialize a :mod:`Pyplot <matplotlib.pyplot>` figure.

        Set its size and dpi, set the font size,
        choose the Arial font family if possible,
        and append it to the figures dictionary.

        Args:
            fig_name:
                name of figure; used as key in figures dictionary
            fig_size:
                optional width and height of figure in inches
            dpi:
                rasterization resolution

        Returns:
            :obj:`Pyplot figure <matplotlib.figure.Figure>`:
                reference to :mod:`MatPlotLib/Pyplot <matplotlib.pyplot>`
                figure
        """
        fig_size_: Tuple[float, float] = (
            (8, 8) if fig_size is None else fig_size
        )
        dpi_: float = self.dpi if dpi is None else dpi
        logging.info(
            "gmplib.plot.GraphingBase:\n   "
            + f"Creating plot: {fig_name} size={fig_size_} @ {dpi_} dpi"
        )
        if sub_plots is None:
            fig = plt.figure()
        else:
            fig, _ = plt.subplots(*sub_plots)
            logging.info(
                f"   with {sub_plots} sub-plots"
            )
        self.fdict.update({fig_name: fig})
        fig.set_size_inches(fig_size_)
        fig.set_dpi(dpi_)
        return fig

    def plot_model(
            self, 
            name: str,
            graph: Graph,
            communities: Communities,
            geometry: Geometry,
            community: Optional[Any] = None, #np.lib.index_tricks.IndexExpression]
            fig_size: Optional[Tuple[float, float]] = None,
            dpi: Optional[int] = None,
        ) -> None:
        r"""
        Plot model geometry in 2D.

        Arguments:
            name: reference for figure dictionary
            mg: model geometry
            community: numpy slice of subset of clique communities to plot
            fig_size: (optional) x,y dimensions of figure
            dpi: (optional) plot resolution
        """
        # _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        fig = self.create_figure(fig_name=f"{name}_mesh", fig_size=fig_size, dpi=dpi)
        d_node_vertices = graph.d_node_vertices
        communities_triangles = \
            communities.d_community_triangles if community is None \
            else np.array(communities.d_community_triangles, dtype=np.object_,)[community]
        for keynode_ in geometry.d_keynode_communities:
            plt.plot(*d_node_vertices[keynode_][0:2], "o", color="gray",)
        for ckey_, community_triangles_ in communities_triangles.items():
            c_ = "k" if ckey_== geometry.groundcommunity else color(ckey_) 
            for triangle_ in community_triangles_:
                triangle_vertices_ = np.array([
                    d_node_vertices[node_][0:2]
                    for node_ in (loop(triangle_))
                ])
                plt.plot(*(triangle_vertices_.T), "o", c=c_, ms=2,)
                plt.fill(*(triangle_vertices_.T), "-", c=c_, alpha=0.3,)
                plt.plot(*(triangle_vertices_.T), "-", c=c_, alpha=1, lw=0.5,)
        gca = fig.gca()
        gca.set_aspect(1)
        plt.grid(":", alpha=0.3)

    def build_mesh(
            self, 
            geometry: Geometry,
        ) -> PVMesh:
        communities = geometry.cmnts
        graph = communities.graph
        faces = (
            [[3]+list(nodes_) for nodes_ in graph.d_triangle_trinodes.values()]
        )
        pvmesh: PVMesh = pv.PolyData(graph.vertices, faces)
        pvmesh.cell_data["colors"] = np.zeros([graph.n_triangles,3])
        for face_, triangles_nodes_ in communities.d_community_triangles.items():
            for triangle_nodes_ in triangles_nodes_:
                triangle_ = graph.d_trinodes_triangles[triangle_nodes_]
                pvmesh.cell_data["colors"][triangle_] = (
                    to_rgb("#d0d0d0") if face_== geometry.groundcommunity else
                    to_rgb(color(face_))
                )
        return pvmesh
    
    def plot_model_3d(
            self,
            pvmesh: PVMesh,
            do_show_edges: Optional[bool] = True,
            do_lighting: Optional[bool] = False,
            do_trilabels = False,
            backend: str = "trame"
        ) -> PVPlotter:
        p = pv.Plotter(notebook="true",)
        _ = p.add_mesh(
            pvmesh,
            scalars="colors",
            lighting=do_lighting,
            rgb=True,
            show_edges=do_show_edges,
            preference="cell",
        )
        # points = pvmesh.points
        # mask = points[:, 0] == 0
        # p.add_point_labels(points[mask], points[mask].tolist(), point_size=20, font_size=36)
        # p.add_point_scalar_labels(pvmesh, "colors", point_size=20, font_size=36)
        cell_labels = [f'{i}' for i in range(pvmesh.n_cells)]
        if do_trilabels:
            p.add_point_labels(pvmesh.cell_centers(), cell_labels, font_size=10)
        p.camera_position = "xy"
        p.show(jupyter_backend=backend)
        return p
    
    def plot_graph(
            self,
            name: str,
            graph: Graph,
            fig_size: Optional[Tuple[float, float]] = None,
            dpi: Optional[int] = None,
        ) -> None:
        r"""
        Plot graph of mesh.

        Arguments:
            name: reference for figure dictionary
            graph: networkx graph of mesh
            fig_size: (optional) x,y dimensions of figure
            dpi: (optional) plot resolution
        """
        _ = self.create_figure(fig_name=f"{name}_graph", fig_size=fig_size, dpi=dpi)
        nx.draw_kamada_kawai(graph.nxgraph, node_size=20,)