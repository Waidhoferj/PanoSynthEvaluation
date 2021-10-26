from argparse import ArgumentParser
import os
import glob


def extract_key(p: str) -> int:
    """
    extracts number from filename formatted like `file_01.suffix`
    """
    return int(os.path.basename(p).split("_")[1].split(".")[0])


def batch_rename(cur_filenames: list, new_filenames: list):
    """
    renames a group of files while avoiding name conflicts
    """
    path, f = os.path.split(cur_filenames[0])
    _, suffix = os.path.splitext(f)
    tmp_names = [os.path.join(path, str(i) + suffix) for i in range(len(cur_filenames))]
    for source, tmp in zip(cur_filenames, tmp_names):
        os.rename(source, tmp)
    for tmp, dest in zip(tmp_names, new_filenames):
        os.rename(tmp, dest)


def rename_snapshots(directory: str):
    directory = os.path.relpath(directory)
    poses = glob.iglob(os.path.join(directory, "**", "poses"), recursive=True)
    snapshots = glob.iglob(os.path.join(directory, "**", "snapshots"), recursive=True)
    # update snapshot names
    for snap_path in snapshots:
        source_names = sorted(glob.glob(os.path.join(snap_path, "*.png")))
        dest_names = sorted(source_names, key=extract_key)
        batch_rename(source_names, dest_names)
    # update pose names
    for pose_path in poses:
        source_names = sorted(glob.glob(os.path.join(pose_path, "*.json")))
        dest_names = sorted(source_names, key=extract_key)
        batch_rename(source_names, dest_names)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = args.directory
    rename_snapshots(directory)
