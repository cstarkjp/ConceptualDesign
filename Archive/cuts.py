vpoints_list = [tuple(vpoint_) for vpoint_ in (np.round(graph.vpoints,3))]
vpoints_set = frozenset(vpoints_list)
# [vpoint_ for vpoint_ in vpoints_set if vpoint_ in vpoints_list]
# if vpoints_list!=vpoints_set:
#     [
#         vertex_,vpoints_ 
#         for vertex_,vpoints_ in graph.d_vertex_vpoints.items()
#         if 
#     ]

vpoints_multiset = Counter()
for vpoint_ in ((graph.vpoints)):
    vpoints_multiset.update([tuple(vpoint_)])
pp(vpoints_multiset)













# def sum_Fk(Fk_set):
#     # Fk_set = frozenset([Fk_ for Fk_ in Fk_set])
#     Fk_sum = None
#     for Fk_ in  Fk_set:
#         Fk_sum = Fk_ if Fk_sum is None else Fk_sum+Fk_
#     return Fk_sum

# sum_Fk(d_node_Fk.values())

# d_node_Fk = {
#     node_: MatrixSymbol(rf"F_{node_}",2,1)
#     for node_ in nodes_
# }
# frozenset(d_node_Fk.values())



    # def plot_raw_model(self, mg: Topology,) -> None:
    #     fig = self.create_figure(fig_name=f"mesh", fig_size=(8,8,),)
    #     m = mg.mesh
    #     tm = mg.trimesh
    #     for i_, (triangle_, v0_, v1_, v2_,) in enumerate(zip(
    #             m.vectors, m.v0[:,0:2], m.v1[:,0:2], m.v2[:,0:2],
    #         )):
    #         looptri_ = np.vstack([triangle_[:,0:2],triangle_[0,0:2]]).T
    #         plt.fill(*looptri_, "-", c=color(i_), lw=1, alpha=0.3,)
    #         plt.plot(*looptri_, "-", c=color(i_), lw=1, alpha=1,)
    #     for i_, v_ in enumerate(tm.vpoints[:,0:2]):
    #         plt.plot(*v_,"ok", ms=2,)
    #     gca = fig.gca()
    #     gca.set_aspect(1)
    #     plt.grid(":", alpha=0.3)









        # self.triangle_areas: NDArray = np.array([
        #     area(self.chop(np.array([
        #         self.d_vertex_vpoints[vertex_]
        #         for vertex_ in triangle_
        #     ])))
        #     for triangle_ in self.d_triangle_trivertices.values()
        # ])    

  
    # def build_nodes_dict(self):
    #     # This is not an efficient algorithm!
    #     # Worse: it fails to count *all* the connected communities per node
    #     for community_, vertices_ in self.d_community_vertices.items():
    #         d_othercommunity_vertices: Dict = self.d_community_vertices.copy()
    #         del d_othercommunity_vertices[community_]
    #         # Now we have (1) a target community(2) the other communities
    #         # Step through the other communities
    #         for othercommunity_, othervertices_ in d_othercommunity_vertices.items():
    #             # Search through all the vertices of the target community
    #             for vertex_ in vertices_:
    #                 if vertex_ in othervertices_:
    #                     yield(vertex_, (community_,othercommunity_))


from operator import add


from functools import reduce
import pyvista as pv
from stl import mesh
        # self.mesh = mesh.Mesh.from_file(self.file_path_name)
        # reader = pv.get_reader(self.file_path_name)
        # self.pvmesh = reader.read()

plotter = pv.Plotter()
plotter.add_mesh(mg.pvmesh, show_edges=True, color='r', ) #scalars="colors",)
# plotter.disable_anti_aliasing()
plotter.view_xy()
plotter.show(jupyter_backend="trame")

    
    # def sort_triangles_by_area(self):
    #     self.sorted_triangle_indexes: NDArray = np.r_[list(np.flip(
    #         np.argsort(self.triangle_areas)
    #     ))]
    #     self.sorted_triangles: NDArray = (np.array(self.d_triangle_trivertices)[self.sorted_triangle_indexes])


    # def find_ground_vpoints(self):
    #     chopped_vpoints: NDArray = self.chop(self.vpoints)[:,0:2]
    #     self.ground_vpoints: NDArray = np.argwhere(chopped_vpoints[:,1]==0).flatten()       


# import pyvista as pv
# pv.set_plot_theme("document")
# pv.set_jupyter_backend("trame")

# # cpos = [(-0.08566, 0.18735, 0.20116), (-0.05332, 0.12168, -0.01215), (-0.00151, 0.95566, -0.29446)]
# reader = pv.get_reader(mg.file_path_name)
# pvmesh = reader.read()
# # mesh.plot()
# pl = pv.Plotter()
# pl.add_mesh(pvmesh, show_edges=True)
# # pl.disable_anti_aliasing()
# # pl.camera_position = cpos
# pl.show(jupyter_backend='trame')

# fig = viz.create_figure(fig_name=f"structure", fig_size=(6,6,),)
# axes = fig.add_subplot(projection='3d')
# poly3d_collection = mplot3d.art3d.Poly3DCollection(mg.mesh.vectors)
# axes.add_collection3d(poly3d_collection)
# scale = mg.mesh.points.flatten()
# axes.auto_scale_xyz(scale, scale, scale)
# axes.view_init(elev=90, azim=0, roll=90)
# # axes.set_zlim3d(0, 0)
# # axes.set_axis_off()
# axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# axes.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# axes.set_zticks([]);

# fig = viz.create_figure(fig_name=f"trimesh", fig_size=(8,8,),)
# for v_ in tm.vpoints[:,0:2]:
#     # looptri_ = np.vstack([triangle_[:,0:2],triangle_[0,0:2]]).T
#     # plt.fill(*looptri_, "-", c=color(i_), lw=1, alpha=0.1,)
#     # plt.plot(*looptri_, "-", c=color(i_), lw=1, alpha=1,)
#     plt.plot(*v_,"o", c=color(i_), ) #alpha=0.2, ms=12,)
#     # plt.plot(*v1_,"^", c=color(i_), ms=9, alpha=0.4)
#     # plt.plot(*v2_,"s", c=color(i_), ms=5, )
#     # plt.plot(v0_[0],v1_[1],".", c=color(i_),)
#     # break
# gca = fig.gca()
# gca.set_aspect(1)
# plt.grid(":", alpha=0.3)

# for facet in tm.facets:
#     tm.visual.face_colors[facet] = trimesh.visual.random_color()
# tm.show()


# def plot_model(m, tm):
#     fig = viz.create_figure(fig_name=f"mesh", fig_size=(8,8,),)
#     for i_, (triangle_, v0_, v1_, v2_,) in enumerate(zip(m.vectors, m.v0[:,0:2], m.v1[:,0:2], m.v2[:,0:2],)):
#         looptri_ = np.vstack([triangle_[:,0:2],triangle_[0,0:2]]).T
#         plt.fill(*looptri_, "-", c=color(i_), lw=1, alpha=0.3,)
#         plt.plot(*looptri_, "-", c=color(i_), lw=1, alpha=1,)
#         # plt.plot(*v0_,"o", c=color(i_), alpha=0.2, ms=12,)
#         # plt.plot(*v1_,"^", c=color(i_), ms=9, alpha=0.4)
#         # plt.plot(*v2_,"s", c=color(i_), ms=5, )
#         # plt.plot(v0_[0],v1_[1],".", c=color(i_),)
#         # break
#     for i_, v_ in enumerate(tm.vpoints[:,0:2]):
#         # looptri_ = np.vstack([triangle_[:,0:2],triangle_[0,0:2]]).T
#         # plt.fill(*looptri_, "-", c=color(i_), lw=1, alpha=0.1,)
#         # plt.plot(*looptri_, "-", c=color(i_), lw=1, alpha=1,)
#         plt.plot(*v_,"ok", ) #c=color(i_), alpha=1,)
#         # plt.plot(*v1_,"^", c=color(i_), ms=9, alpha=0.4)
#         # plt.plot(*v2_,"s", c=color(i_), ms=5, )
#         # plt.plot(v0_[0],v1_[1],".", c=color(i_),)
#         # break
#     gca = fig.gca()
#     gca.set_aspect(1)
#     plt.grid(":", alpha=0.3)