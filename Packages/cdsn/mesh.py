"""
Class to build a mesh topology from a 3d data file.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`trimesh`

---------------------------------------------------------------------
"""
# Library
import warnings
import logging
from typing import (
    Dict, Any, Tuple, Optional, List, Callable, Iterable, Sized, Generator,
)

import os
import trimesh

from trimesh.visual import texture, TextureVisuals
from trimesh import Trimesh

from cdsn.definitions import (
    NDArray, Trimesh, TrimeshTrackedArray, NXGraph
)

warnings.filterwarnings("ignore")

__all__ = ["Mesh"]

class Mesh:
    """
    Class to build a mesh from a 3D data export file.

    Args:
        name (str):
            name of source mesh file (without file extension)
        data_path (optional str):
            relative path from here to data file (default: ../Data/Export/)
        file_type (optional str):
            type of 3D export file (default: "STL"; or one of "3MF", "ThreeMF", "glTF", "Collada", "DAE")

    Attributes:
        file_path_name (str): 
            relative path to STL file and its name with ".stl" extension
        trimesh (Trimesh):
            triangular mesh imported from STL file in Trimesh format
    """
    def __init__(
            self,
            name: str,
            data_path: str = os.path.join(os.pardir,"Data","Export",),
            file_type: str = "stl",
        ) -> None:
        # Read model from STL or other format file
        self.name = name
        self.read_from_file(data_path, name, file_type,)

    def read_from_file(
            self,
            data_path: str,
            name: str,
            file_type: str,
        ) -> None:
        """
        Use Trimesh to load an 3D export file.

        Args:
            data_path (str):
                relative path to the data folder containing the data file
            name (str):
                data file name
            file_type (str):
                type of 3D export file

        Attributes:
            file_path_name (str):
                relative path and name of data file
            trimesh (Trimesh):
                processed mesh as a Trimesh object
        """
        # Collada doesn't work yet. Gives error in graph():
        #  AttributeError: 'Scene' object has no attribute 'edges_unique'
        d_file_types = {
            "stl": "stl",
            "dae": "dae",
            "collada": "dae",
            "obj": "obj",
            "gltf": "gltf",
            "3mf": "3mf",
            "threemf": "3mf",
        }
        self.file_type = d_file_types[file_type.lower()]
        self.file_path_name: str = os.path.join(data_path,f"{name}.{self.file_type}")
        self.trimesh: Trimesh = trimesh.load(
            self.file_path_name, 
            # process=(True if self.file_type=="stl" else False), 
            process=True,
            force="mesh",
        )
        self.trimesh.merge_vertices(merge_tex=True,)

        # If this is a glTF and we have metadata available:
        self.gltf: Optional[Dict] = None
        self.d_gltf_json: Optional[Dict] = None
        self.d_glnode_info: Optional[Dict] = None
        if self.file_type=="gltf":
            from pygltflib import GLTF2
            self.gltf = GLTF2().load(self.file_path_name)
            # self.d_glnode_info: Dict = {
            #     glnode_.mesh: glnode_.name 
            #     for glnode_ in self.gltf.nodes
            # }
            import json
            with open(self.file_path_name, "r") as file:
                self.d_gltf_json = json.load(file)
            self.d_glnode_info = {
                glnode_["mesh"]: glnode_["name"] 
                for glnode_ in self.d_gltf_json["nodes"]
            }
        else:
            self.gltf = None
            self.d_gltf_json = None
            self.d_glnode_info = None
 
    # def parse(self) -> None:
    #     self.member_info: Dict = {
    #         member_.mesh: member_.name 
    #         for member_ in self.gltf.nodes
    #     }
        


# from trimesh.visual import texture, TextureVisuals
# from trimesh import Trimesh

# def get_texture(my_uvs, img):
#     # img is PIL Image
#     uvs = my_uvs
#     material = texture.SimpleMaterial(image=img)    
#     texture = TextureVisuals(uv=uvs, image=img, material=material)

# my_uvs = [....] # 2d array list
# vertices = [....] # 3d array list
# faces = [....] # indices list
# face_normals = [....] # 3d array list
# texture_visual = get_texture(my_uvs, img)
# mesh = Trimesh(
#             vertices=vertices,
#             faces=faces,
#             face_normals=face_normals,
#             visual=texture_visual,
#             validate=True,
#             process=False
#         )