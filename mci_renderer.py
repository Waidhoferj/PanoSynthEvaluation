from typing import Tuple
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from rendertools import *


def compute_sigma(pred_disp, actual_disp):
    """
    Computes an average scale different between true and predicted disparity.
    """
    zero_mask = (actual_disp != 0) & (pred_disp != 0)
    actual_disp = np.log(actual_disp[zero_mask])
    pred_disp = np.log(pred_disp[zero_mask])
    sigma = np.exp(np.median(pred_disp - actual_disp))
    return sigma


class MCIRenderer(Renderer):
    def __init__(self, image_size: Tuple[int, int], texture_path: str, sigma=1):
        min_depth = 1.0
        max_depth = 100.0
        depths = 1.0 / np.linspace(1.0 / max_depth, 1.0 / min_depth, 32, endpoint=True)

        depths = [i * sigma for i in depths]
        meshes = [
            Cylinder(
                bottom=-1 * depth,
                top=1 * depth,
                radius=depth,
                texturepath=texture_path % i,
            )
            for i, depth in enumerate(depths)
        ]
        super().__init__(meshes, image_size[1], image_size[0], offscreen=True)
        self.fovy = 90

    def render_image(self, eye: np.ndarray, target: np.ndarray, up=np.array([0, 1, 0])):
        """
        Generates an snapshot using the MCI renderer
        - `eye`: location of the camera
        - `target`: center of the camera's view
        - `up`: up direction from the camera's perpective
        """

        view_matrix = lookAt(eye, target, up)
        proj_matrix = perspective(self.fovy, self.width / self.height, 0.1, 1000.0)
        mvp_matrix = proj_matrix @ view_matrix
        produced_image = self.render(mvp_matrix)
        return produced_image
