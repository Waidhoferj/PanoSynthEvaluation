from OpenGL.GL import *
from OpenGL.GLUT import *
import time

import numpy as np
import os
from rendertools import *

from imageio import imwrite

import tensorflow as tf
from pathlib import Path
from habitat.panorama_extractor import PanoExtractor
from evaluation import render_image, compute_sigma
import subprocess
from habitat.convert import Converter 
import imageio
import subprocess
import pysvn
import requests
import tarfile




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
    img_size = (512,512)
    pano_size = (img_size[0], img_size[0] * 2)

    check_dependencies()
    # Import after dependencies exist
    from generate_mpi import generate_mpi


    extractor = PanoExtractor(
        scene_path,
        img_size=pano_size,
        output=["rgba", "depth"],
        shuffle=False)

    sphere_pano = extractor[0]
    
    converter = Converter(output_height=pano_size[0],
                          output_width=pano_size[1])
    os.makedirs(cylinder_path, exist_ok=True)
    depth = sphere_pano["depth"]
    depth[depth < 1.0] = 1.0
    depth = 255.0 / depth
    depth = depth.astype("uint8")
    for filename, pano in zip(["scene.jpeg", "actual_depth.png"], [sphere_pano["rgba"][..., :3], depth]):
        if len(pano.shape) < 3:
            pano = np.expand_dims(pano, axis=-1)
        res = converter.convert(pano).numpy().astype("uint8")
        imwrite(os.path.join(cylinder_path,filename), res)

    disparity_map, layers = generate_mpi(os.path.join(cylinder_path, "scene.jpeg"))
    imwrite(os.path.join(cylinder_path, "predicted_depth.png"), disparity_map)
    os.makedirs("data/layers", exist_ok=True)
    for i, layer in enumerate(layers):
        imageio.imsave(f"data/layers/layer_{i}.png", layer.numpy())


    # Create output directory
    os.makedirs("data/renderer-comparison", exist_ok=True)

    # Calculate sigma
    predicted_depth = imread(os.path.join(
        cylinder_path, 'predicted_depth.png')).astype('float32')
    # interpolate depths that are far too close together
    predicted_depth= (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min()) * 255
    
    actual_depth = imread(os.path.join(
        cylinder_path, 'actual_depth.png')).astype('float32')
    sigma = 1 # TODO: Fix issue with compute_sigma
    # compute_sigma(predicted_depth, actual_depth).numpy()
    angles = np.linspace(0, 2* np.pi, 13)
    targets = [np.array([np.sin(angle), 0, np.cos(angle)]) for angle in angles]
    eye = np.zeros(3)
    

   
    
    habitat_renders = [extractor.create_snapshot(0,target)[..., :3] for target in targets]
    extractor.close()

    rot = np.pi / 180 * 210
    translation = np.array([1,0,30])
    eye += translation
    
    transformed_angles = [angle + rot for angle in angles]
    targets = [np.array([np.sin(angle), 0, np.cos(angle)]) for angle in transformed_angles]
    targets += translation

    # Create mci renders
    mci_renders = [render_image(img_size, texture_path, eye, target, sigma=sigma) for target in targets]

    for angle, mci_render, habitat_render in zip(angles, mci_renders, habitat_renders):
        image = np.concatenate([mci_render, habitat_render], axis=1)
        imwrite(f'data/renderer-comparison/comparison_{round(angle * 180.0 / np.pi)}.png', image)
        mse = ((mci_render - habitat_render)**2).mean(axis=None)
        print('MSE: ', mse)
    # shutil.rmtree("data")
