from argparse import ArgumentParser
from shutil import rmtree
from glob import iglob
from os import path


def delete_layers(data_dir: str):
    i = 0
    for layer_dir in iglob(path.join(data_dir, "**", "layers"), recursive=True):
        rmtree(layer_dir)
        i += 1
    print(f"Removed {i} layer folders.")


if __name__ == "__main__":

    parser = ArgumentParser(description="Deletes all layer images from a `data` folder")
    parser.add_argument("--folder", "-f", default="data", help="path to data folder")
    args = parser.parse_args()
    delete_layers(args.folder)
