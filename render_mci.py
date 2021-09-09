"""
Functions that construct the MCI environment and extract images from it.
"""

from OpenGL.GL import *
from OpenGL.GLUT import *

import numpy as np
import os
import re

from rendertools import *


import json

import argparse
from typing import Tuple


def compute_sigma(pred_disp, actual_disp):
    """
    Computes an average scale different between true and predicted disparity.
    """
    zero_mask = (actual_disp != 0) & (pred_disp != 0)
    actual_disp = np.log(actual_disp[zero_mask])
    pred_disp = np.log(pred_disp[zero_mask])
    sigma = np.exp(np.median(pred_disp - actual_disp))
    return sigma


# python evaluation.py cube_dir/castle/ cube_dir/castle/00/test/snapshot_0.png
renderer = None
window = None


def render_image(
    image_size: Tuple[int, int],
    texture_path: str,
    eye: np.ndarray,
    target: np.ndarray,
    up=np.array([0, 1, 0]),
    sigma=1,
):
    """
    Generates an snapshot using the MCI renderer
    - `image_size`: height by width of the image
    - `texture_path`: templated string that points to the MCI layer textures
    - `eye`: location of the camera
    - `target`: center of the camera's view
    - `up`: up direction from the camera's perpective
    - `sigma`: scaling factor applied to the concentric cylinders
    """
    global renderer
    global window
    height, width = image_size
    fovy = 90
    if not window:
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_3_2_CORE_PROFILE)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("window")
        glutHideWindow(window)
    min_depth = 1.0
    max_depth = 100.0
    depths = 1.0 / np.linspace(1.0 / max_depth, 1.0 / min_depth, 32, endpoint=True)

    depths = [i * sigma for i in depths]
    meshes = [
        Cylinder(
            bottom=-1 * depth, top=1 * depth, radius=depth, texturepath=texture_path % i
        )
        for i, depth in enumerate(depths)
    ]

    renderer = Renderer(meshes, width=width, height=height, offscreen=True)

    view_matrix = lookAt(eye, target, up)
    proj_matrix = perspective(fovy, width / height, 0.1, 1000.0)
    mvp_matrix = proj_matrix @ view_matrix

    produced_image = renderer.render(mvp_matrix)
    return produced_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cylinder_path")
    parser.add_argument("test_path")
    args = parser.parse_args()

    target_image = imread(args.test_path)

    regexr = "\/(\w+)\/(\w+).png"
    folder, image_name = re.findall(regexr, args.test_path)[0]
    target_json_path = os.path.join(
        args.test_path.split(folder)[0],
        f"{folder}_json",
        image_name.split("_frame")[0] + ".json",
    )

    with open(target_json_path) as f:
        target_json = json.load(f)

    # Scale Depths
    predicted_depth = imread(
        os.path.join(args.cylinder_path, "predicted_depth.png")
    ).astype("float32")
    actual_depth = imread(os.path.join(args.cylinder_path, "actual_depth.png")).astype(
        "float32"
    )
    sigma = compute_sigma(predicted_depth, actual_depth).numpy()

    eye = np.array([target_json["eye"]])
    target = np.array(target_json["target"])
    target[2] = -target[2]
    up = np.array(target_json["up"])

    produced_image = render_image(
        target_image.shape[:2],
        os.path.join(args.cylinder_path, "layers/layer_%d.png"),
        eye,
        target,
        up,
    )

    # Calculate MSE HERE
    mse = ((produced_image - target_image[..., :3]) ** 2).mean(axis=None)
    print("MSE: ", mse)
    sys.exit(0)
