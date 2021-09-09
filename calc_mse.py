from argparse import ArgumentParser
import os
import glob
from posix import listdir
from posixpath import join
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate the MSE")

    parser.add_argument("--data-dir", help="data directory", required=True)

    args = parser.parse_args()
    mse_vals = []
    for location in glob.iglob(os.path.join(args.data_dir, "**", "location_*")):
        actual_paths = glob.iglob(os.path.join(location, "snapshots", "*.png"))
        predicted_paths = glob.iglob(
            os.path.join(location, "predicted-snapshots", "*.png")
        )
        for actual_path, predicted_path in zip(actual_paths, predicted_paths):
            actual, predicted = (
                plt.imread(actual_path)[..., :3],
                plt.imread(predicted_path)[..., :3],
            )
            mse = np.mean(actual - predicted)
            mse_vals.append(mse)

    for mse in mse_vals:
        print(f"mse = {mse}")
    print(f"\navg mse {np.mean(mse_vals)}")
