from argparse import ArgumentParser
from shutil import rmtree
from glob import iglob
from os import path
import imageio
import numpy as np
import os


def find_min_depth(data_dir: str):
    max_disp = -np.inf
    for location in iglob(
        path.join(data_dir, "**", "actual_disparity.npy"), recursive=True
    ):
        disp = np.load(location).max()
        if disp > max_disp:
            max_disp = disp
    z_min = 1.0 / max_disp
    print(f"z_min = {z_min}")


if __name__ == "__main__":

    parser = ArgumentParser(description="Finds the min depth in the dataset")
    parser.add_argument("--folder", "-f", default="data", help="path to data folder")
    args = parser.parse_args()
    find_min_depth(args.folder)
