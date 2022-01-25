from argparse import ArgumentParser
from shutil import rmtree
from glob import iglob
from os import path
import imageio
import numpy as np
import os


def has_rip(image_path: str) -> bool:
    thresh = 0.25
    im = imageio.imread(image_path)
    total_pixels = np.prod(im.shape[:2])
    return np.sum(np.mean(im, axis=-1) == 0) > (thresh * total_pixels)


def filter_out_blanks(data_dir: str):
    i = 0
    pose_offset = len("pose_")
    snapshot_offset = len("snapshot_")
    for location in iglob(path.join(data_dir, "**", "location_*"), recursive=True):

        # print(f"Scanning {location}...")
        dark_images = set()
        mci_snaps = iglob(path.join(location, "mci-snapshots", "*.png"))
        mesh_snaps = iglob(path.join(location, "mesh-snapshots", "*.png"))
        poses = iglob(path.join(location, "poses", ".json"))
        for snap in list(mci_snaps) + list(mesh_snaps):
            if has_rip(snap):
                dark_images.add(path.basename(snap))
                i += 1
        for img_base in dark_images:
            for img in iglob(path.join(location, "*", img_base)):
                os.remove(img)

    print(f"found {i} dark shots with this threshold.")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Deletes all blacked out images from a `data` folder"
    )
    parser.add_argument("--folder", "-f", default="data", help="path to data folder")
    args = parser.parse_args()
    filter_out_blanks(args.folder)
