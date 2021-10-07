"""
DEPRECATED
TEST FILE
Compares the coordinate systems of the MCI renderer and the CylinderExtractor
"""

from OpenGL.GL import *
from OpenGL.GLUT import *
import shutil

import numpy as np
import os
from rendertools import *
import glob

from imageio import imwrite

from scipy.spatial.transform import Rotation
from habitat.cylinder_extractor import CylinderExtractor
from render_mci import render_image, compute_sigma
import imageio
import json

from utils import check_dependencies


if __name__ == "__main__":
    # Filepaths
    cylinder_path = "data/cylinder-panos"
    scene_path = (
        "habitat/scenes/skokloster-castle.glb"
    )  # TODO: automatically download scenes into folder
    texture_path = os.path.join("data", "layers", "layer_%d.png")

    # Parameters
    # Will run Habitat, panorama conversion, and MPI model pipeline if True. Otherwise uses previously produced images.
    should_generate_data = True
    img_size = (512, 512)
    pano_size = (img_size[0], img_size[0] * 4)
    pano_index = 3
    sample_count = 10

    check_dependencies()
    # Import after dependencies exist
    from generate_mpi import generate_mpi

    if should_generate_data:
        # Erase old data entries
        if os.path.exists("data"):
            shutil.rmtree("data")
        os.makedirs(cylinder_path, exist_ok=True)
        extractor = CylinderExtractor(
            scene_path,
            img_size=(512, 2048),
            output=["rgba", "depth"],
            pose_extractor_name="cylinder_pose_extractor",
            shuffle=False,
        )

        cylinder_pano = extractor.create_panorama(pano_index)
        depth = cylinder_pano["depth"]
        depth = depth.astype("uint8")
        imwrite(os.path.join(cylinder_path, "actual_depth.png"), depth)
        imwrite(
            os.path.join(cylinder_path, "scene.jpeg"),
            cylinder_pano["rgba"].astype("uint8")[..., :3],
        )

        disparity_map, layers = generate_mpi(os.path.join(cylinder_path, "scene.jpeg"))
        imwrite(os.path.join(cylinder_path, "predicted_depth.png"), disparity_map)
        os.makedirs("data/layers", exist_ok=True)
        for i, layer in enumerate(layers):
            imageio.imsave(
                f"data/layers/layer_{i}.png", (layer.numpy() * 255).astype("uint8")
            )
        os.makedirs("data/snapshots", exist_ok=True)
        for i in range(10):
            snapshot, pose = extractor.random_snapshot(pano_index)
            with open(f"data/snapshots/pose_{i}.json", "w") as f:
                json.dump(pose, f)
            imageio.imsave(f"data/snapshots/snapshot_{i}.png", snapshot)

        extractor.close()

    # Create output directory
    os.makedirs("data/renderer-comparison", exist_ok=True)

    # Calculate sigma
    predicted_depth = imread(os.path.join(cylinder_path, "predicted_depth.png")).astype(
        "float32"
    )
    # interpolate depths that are far too close together
    predicted_depth = (
        (predicted_depth - predicted_depth.min())
        / (predicted_depth.max() - predicted_depth.min())
        * 255
    )

    actual_depth = imread(os.path.join(cylinder_path, "actual_depth.png")).astype(
        "float32"
    )
    sigma = compute_sigma(predicted_depth, actual_depth)
    # Load habitat snapshot renders
    snapshotPaths = sorted(glob.glob("data/snapshots/*.png"))
    habitat_renders = [imageio.imread(path)[..., :3] for path in snapshotPaths]

    # Apply all coordinate transformations that align habitat to MCI to the `transform` matrix
    transform = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    poses = []
    posePaths = sorted(glob.glob("data/snapshots/*.json"))
    for path in posePaths:
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
    mci_renders = [
        render_image(img_size, texture_path, eye, target, up=up, sigma=sigma)
        for eye, target, up in poses
    ]
    # compare renderer outputs
    i = 0
    for habitat_render, mci_render in zip(habitat_renders, mci_renders):
        image = np.concatenate([habitat_render, mci_render], axis=1)
        imwrite(f"data/renderer-comparison/comparison_{i}.png", image)
        mse = ((mci_render - habitat_render) ** 2).mean(axis=None)
        print("MSE: ", mse)
        i += 1
