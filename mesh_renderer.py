from typing import Tuple
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from rendertools import *


class DepthMeshRenderer(Renderer):
    def __init__(
        self, image_size: Tuple[int, int], scene_texture_path: str, disparity_path: str
    ):
        scale = 3
        meshes = [
            DepthCylinder(
                height=5 * scale,
                radius=scale,
                texturepath=scene_texture_path,
                disparitypath=disparity_path,
                nsegments=360,
                nvertsegments=63,
            )
        ]
        super().__init__(meshes, image_size[1], image_size[0], offscreen=True)
        self.fovy = 90

    def render_image(self, eye: np.ndarray, target: np.ndarray, up=np.array([0, 1, 0])):
        """
        Generates an snapshot using the Mesh renderer
        - `eye`: location of the camera
        - `target`: center of the camera's view
        - `up`: up direction from the camera's perpective
        """

        view_matrix = lookAt(eye, target, up)
        proj_matrix = perspective(self.fovy, self.width / self.height, 0.1, 1000.0)
        mvp_matrix = proj_matrix @ view_matrix
        produced_image = self.render(mvp_matrix)
        return produced_image
