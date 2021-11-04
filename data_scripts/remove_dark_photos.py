from argparse import ArgumentParser
from shutil import rmtree
from glob import iglob
from os import path
import imageio
import numpy as np


def has_rip(image_path: str) -> bool:
    thresh = 20
    im = imageio.imread(image_path)
    return np.sum(im == 0) > thresh


def filter_out_blanks(data_dir: str):
    i = 0
    pose_offset = len("pose_")
    snapshot_offset = len("snapshot_")
    for location in iglob(path.join(data_dir, "**", "location_*"), recursive=True):
        dark_images = set()
        mci_snaps = iglob(path.join(location, "mci-snapshots", "*.png"))
        mesh_snaps = iglob(path.join(location, "mesh-snapshots", "*.png"))
        poses = iglob(path.join(location, "poses", ".json"))
        # For all
        i += 1
    print(f"Removed {i} layer folders.")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Deletes all blacked out images from a `data` folder"
    )
    parser.add_argument("--folder", "-f", default="data", help="path to data folder")
    args = parser.parse_args()
    filter_out_blanks(args.folder)
