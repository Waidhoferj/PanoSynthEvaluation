from OpenGL.GL import *
from OpenGL.GLUT import *
import time

import numpy as np
import os
from rendertools import *

from imageio import imwrite

import tensorflow as tf
from scipy.spatial.transform import Rotation
from pathlib import Path
from habitat.panorama_extractor import PanoExtractor
from habitat.cylinder_extractor import CylinderExtractor
from evaluation import render_image, compute_sigma
import subprocess
from habitat.convert import Converter 
import imageio
import subprocess
import pysvn
import requests
import tarfile
import matplotlib.pyplot as plt




def check_dependencies():
    if not os.path.exists("single_view_mpi"):
        pysvn.Client()
        # NOTE: svn is required for now :(
        subprocess.call(["svn", "export", "--force", "https://github.com/google-research/google-research/trunk/single_view_mpi"])
    if not os.path.exists("single_view_mpi_full_keras"):
        url = "https://storage.googleapis.com/stereo-magnification-public-files/models/single_view_mpi_full_keras.tar.gz"
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=".")

if __name__ == '__main__':
    # Filepaths
    cylinder_path = "data/cylinder-panos"
    scene_path = "habitat/scenes/skokloster-castle.glb" # TODO: automatically download scenes into folder
    texture_path = os.path.join("data", "layers", "layer_%d.png")

    # Parameters
    # Will run Habitat, panorama conversion, and MPI model pipeline if True. Otherwise uses previously produced images.
    should_generate_data = True
    img_size = (512,512)
    pano_size = (img_size[0], img_size[0] * 4)
    pano_index = 3
    # NOTE: This method of generating targets is temporarily used to find the change of coordinates matrix.
    angles = np.linspace(0, 2* np.pi, 13) # angles to generate targets
    targets = [np.array([np.sin(angle), 0, np.cos(angle)]) for angle in angles] # Look-at points
    eye = np.zeros(3)

    check_dependencies()
    # Import after dependencies exist
    from generate_mpi import generate_mpi
    os.makedirs(cylinder_path, exist_ok=True)
    if should_generate_data:
        extractor = CylinderExtractor(
            scene_path,
            img_size=(512, 2048),
            output=["rgba", "depth"],
            pose_extractor_name="cylinder_pose_extractor",
            shuffle=False)

        cylinder_pano = extractor.create_panorama(pano_index)
        depth = cylinder_pano["depth"]
        depth[depth < 1.0] = 1.0
        depth = 255.0 / depth
        depth = depth.astype("uint8")
        imwrite(os.path.join(cylinder_path,"actual_depth.png"), depth)
        imwrite(os.path.join(cylinder_path,"scene.jpeg"), cylinder_pano["rgba"].astype("uint8")[..., :3])
        
        # converter = Converter(output_height=pano_size[0],
        #                     output_width=pano_size[1])
        # os.makedirs(cylinder_path, exist_ok=True)
        # depth = sphere_pano["depth"]
        # depth[depth < 1.0] = 1.0
        # depth = 255.0 / depth
        # depth = depth.astype("uint8")
        # for filename, pano in zip(["scene.jpeg", "actual_depth.png"], [sphere_pano["rgba"][..., :3], depth]):
        #     if len(pano.shape) < 3:
        #         pano = np.expand_dims(pano, axis=-1)
        #     res = converter.convert(pano).numpy().astype("uint8")
        #     imwrite(os.path.join(cylinder_path,filename), res)

        disparity_map, layers = generate_mpi(os.path.join(cylinder_path, "scene.jpeg"))
        imwrite(os.path.join(cylinder_path, "predicted_depth.png"), disparity_map)
        os.makedirs("data/layers", exist_ok=True)
        for i, layer in enumerate(layers):
            imageio.imsave(f"data/layers/layer_{i}.png", (layer* 255).astype("uint8"))
        os.makedirs("data/snapshots", exist_ok=True)
        for i,target in enumerate(targets):
            imageio.imsave(f"data/snapshots/snapshot_{i}.png", extractor.create_snapshot(pano_index,target))
            
        extractor.close()


    # Create output directory
    os.makedirs("data/renderer-comparison", exist_ok=True)

    # Calculate sigma
    predicted_depth = imread(os.path.join(
        cylinder_path, 'predicted_depth.png')).astype('float32')
    # interpolate depths that are far too close together
    predicted_depth= (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min()) * 255
    
    actual_depth = imread(os.path.join(
        cylinder_path, 'actual_depth.png')).astype('float32')
    sigma = compute_sigma(predicted_depth, actual_depth)
    
    habitat_renders = [imageio.imread(f"data/snapshots/snapshot_{i}.png")[..., :3] for i in range(0, len(targets))]
    
    # TODO: Find the change of coordinates that translates between Habitat and MCI Renderer
    # transform = Rotation.from_euler("y", 45, degrees=True).as_matrix()
    # targets = [ target @ transform.T for target in targets]

    # Create mci renders
    mci_renders = [render_image(img_size, texture_path, eye, target, sigma=sigma) for target in targets]
    #plot ranges of sigmas based on the MSE
    for angle, habitat_render, mci_render in zip(angles, habitat_renders, mci_renders):
        image = np.concatenate([ habitat_render, mci_render], axis=1)
        imwrite(f'data/renderer-comparison/comparison_{round(angle * 180 / np.pi)}.png', image)
        mse = ((mci_render - habitat_render)**2).mean(axis=None)
        print('MSE: ', mse)
