def monkeypatch_ctypes():
    """
    MacOS workaround to address issue with PyOpenGl: https://github.com/PixarAnimationStudios/USD/issues/1372
    """
    import os
    import ctypes.util

    uname = os.uname()
    if uname.sysname == "Darwin" and uname.release >= "20.":
        real_find_library = ctypes.util.find_library

        def find_library(name):
            if name in {"OpenGL", "GLUT"}:  # add more names here if necessary
                return f"/System/Library/Frameworks/{name}.framework/{name}"
            return real_find_library(name)

        ctypes.util.find_library = find_library


monkeypatch_ctypes()
from typing import Tuple
from OpenGL.GL import *
from OpenGL.GLUT import *
import glob
from scipy.spatial.transform import Rotation
import json

import numpy as np
import os
import shutil
import imageio
from mesh_render.rendertools import *


import cv2

import argparse


renderer = None
window = None


def render_mesh(
    image_size: Tuple[int, int],
    scene_texture_path: str,
    disparity_path: str,
    eye: np.ndarray,
    target: np.ndarray,
    up=np.array([0, 1, 0]),
):
    """
    Generates an snapshot using the mesh renderer
    - `image_size`: height by width of the image
    - `scene_texture_path`: cylindrical panoramic image of the scene
    - `disparity_path`: path to the predicted disparity map of the scene
    - `eye`: location of the camera
    - `target`: center of the camera's view
    - `up`: up direction from the camera's perpective
    """
    global renderer
    global window
    height, width = image_size
    fovy = 90
    scale = 5.0
    if not window:
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_3_2_CORE_PROFILE)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("window")
        glutHideWindow(window)
    meshes = [
        DepthCylinder(
            height=2.0 * scale,
            radius=scale,
            texturepath=scene_texture_path,
            disparitypath=disparity_path,
            nsegments=360,
            nvertsegments=63,
        )
    ]

    renderer = Renderer(meshes, width=width, height=height, offscreen=True)

    view_matrix = lookAt(eye, target, up)
    proj_matrix = perspective(fovy, width / height, 0.1, 1000.0)
    mvp_matrix = proj_matrix @ view_matrix

    produced_image = renderer.render(mvp_matrix)
    return produced_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    locations = (
        os.path.split(path)[0]
        for path in glob.iglob(
            os.path.join(args.path, "*", "*", "predicted_disparity.png")
        )
    )
    for location in locations:
        print(f"Mesh Renderer: Rendering poses for {location}")

        actual_disparity = imread(
            os.path.join(location, "predicted_disparity.png")
        ).astype("float32")
        # Apply all coordinate transformations that align habitat to MCI to the `transform` matrix
        transform = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        poses = []
        pose_paths = sorted(
            glob.glob(os.path.join(location, "poses", "*.json"))
        )  # TODO: sort on pose number instead of string (will fail for pose count > 10)
        for path in pose_paths:
            with open(path) as f:
                p = json.load(f)
                poses.append(
                    (
                        p["eye"] @ transform.T,
                        p["target"] @ transform.T,
                        p["up"] @ transform.T,
                    )
                )

        # Create mci renders

        out_dir = os.path.join(location, "mesh-snapshots")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        mci_renders = (
            render_mesh(
                (512, 512),
                os.path.join(location, "scene.jpeg"),
                os.path.join(location, "predicted_disparity.png"),
                eye,
                target,
                up=up,
            )
            for eye, target, up in poses
        )
        for i, render in enumerate(mci_renders):
            imageio.imsave(os.path.join(out_dir, f"snapshot_{i}.png"), render)
