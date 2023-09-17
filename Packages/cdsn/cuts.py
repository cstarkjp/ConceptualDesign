
from operator import add


from functools import reduce
import pyvista as pv
from stl import mesh
        # self.mesh = mesh.Mesh.from_file(self.file_name)
        # reader = pv.get_reader(self.file_name)
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
    #     self.sorted_triangles: NDArray = (np.array(self.triangles)[self.sorted_triangle_indexes])


    # def find_ground_vertices(self):
    #     chopped_vertices: NDArray = self.chop(self.vertices)[:,0:2]
    #     self.ground_vertices: NDArray = np.argwhere(chopped_vertices[:,1]==0).flatten()       


# import pyvista as pv
# pv.set_plot_theme("document")
# pv.set_jupyter_backend("trame")

# # cpos = [(-0.08566, 0.18735, 0.20116), (-0.05332, 0.12168, -0.01215), (-0.00151, 0.95566, -0.29446)]
# reader = pv.get_reader(mg.file_name)
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
# for v_ in tm.vertices[:,0:2]:
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
#     for i_, v_ in enumerate(tm.vertices[:,0:2]):
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