import glob
import os
import numpy as np
import imageio
from matplotlib import pyplot


fplans = os.path.join("floorplans")


def remove_images():
    paths = sorted(glob.iglob(os.path.join(fplans, "*.png")))
    for path in paths:
        os.remove(path)


def combine_floorplans():
    paths = sorted(glob.iglob(os.path.join(fplans, "*.npy")))
    if len(paths) == 0:
        print("No floorplans found")
    for i in range(0, len(paths), 2):
        fp = np.load(paths[i]).astype("float")
        fp_eroded = np.load(paths[i + 1]).astype("float")
        combined = (fp + fp_eroded) / 2
        pyplot.imsave(os.path.splitext(paths[i])[0] + ".png", combined)


if __name__ == "__main__":
    combine_floorplans()
