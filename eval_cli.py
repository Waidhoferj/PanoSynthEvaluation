"""
Provides primary user interface to the evaluation pipeline.
"""

from utils import check_dependencies, monkeypatch_ctypes

monkeypatch_ctypes()
from PyInquirer import prompt
from rendertools import *
import glob
from imageio import imwrite
from scipy.spatial.transform import Rotation
from habitat.cylinder_extractor import CylinderExtractor
from render_mci import render_image, compute_sigma
import imageio
import json
import shutil
from utils import check_dependencies
import importlib
from argparse import ArgumentParser

generate_mpi = None


def main():
    global generate_mpi

    parser = ArgumentParser(description="Generate an eval dataset for PanoSynthVR")

    parser.add_argument(
        "--scene-dir", help="directory containing .glb files for habitat rendering"
    )
    parser.add_argument("--data-dir", "-o", help="directory for testing data")
    parser.add_argument(
        "--num-locations", default=5, type=int, help="number of locations"
    )
    parser.add_argument(
        "--num-snapshots",
        default=5,
        type=int,
        help="height of output (cylindrical) panorama image",
    )

    args = parser.parse_args()

    check_dependencies()
    generate_mpi = importlib.import_module("generate_mpi").generate_mpi
    if not args.data_dir:
        # Use inquirer
        answers = prompt(
            [
                {
                    "type": "confirm",
                    "name": "should_generate_scenes",
                    "message": "Generate testing data?",
                    "default": True,
                }
            ]
        )
        if answers["should_generate_scenes"]:
            generate_path()
        else:
            preexisting_path()
    elif args.data_dir and args.scene_dir:
        # Generate new data and run eval
        generate_scene_data(
            args.scene_dir, args.data_dir, args.num_locations, args.num_snapshots
        )
        render_mci_snapshots(args.data_dir)
    else:
        # Run evaluation of provided data
        render_mci_snapshots(args.data_dir)


def generate_path():
    is_int = lambda a: a.isdigit() or "Please input an integer"
    questions = [
        {
            "name": "scene_path",
            "type": "input",
            "message": "Path to the scene folder",
            "validate": lambda a: os.path.isdir(a) or f"'{a}' is not a valid directory",
        },
        {
            "name": "output_path",
            "type": "input",
            "message": "Testing data output path",
            "default": "data",
        },
        {
            "name": "location_count",
            "type": "input",
            "message": "Number of locations per scene",
            "default": "5",
            "validate": is_int,
        },
        {
            "name": "snapshot_count",
            "type": "input",
            "message": "Number of snapshots per location",
            "default": "5",
            "validate": is_int,
        },
    ]

    a = prompt(questions)
    generate_scene_data(
        a["scene_path"],
        a["output_path"],
        int(a["location_count"]),
        int(a["snapshot_count"]),
    )


def preexisting_path():
    a = prompt(
        [
            {
                "name": "data_path",
                "type": "input",
                "message": "Path to testing data",
                "default": "./data",
                "validate": lambda a: os.path.isdir(a)
                or f"'{a}' is not a valid directory.",
            }
        ]
    )

    render_mci_snapshots(a["data_path"])


def generate_scene_data(scene_path, output_path, location_count, snapshot_count):
    """
    Generates new panoramas and snapshots from habitat-sim that serve as the source of truth for evaluation.
    - `scene_path`: the directory holding the scene files.
    - `output_path`: the output directory for the generated data.
    - `location_count`: number of panoramas per scene.
    - `snapshot_count`: number of snapshots per location.
    """
    # Erase old data entries
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    scene_paths = glob.glob(
        os.path.join(scene_path, "**", "*.glb"), recursive=True
    ) + glob.glob(os.path.join(scene_path, "**" "*.ply"), recursive=True)
    for scene in scene_paths:
        scene_name = os.path.split(scene)[1].split(".")[0]
        os.makedirs(os.path.join(output_path, scene_name))
        extractor = CylinderExtractor(
            scene,
            img_size=(512, 2048),
            output=["rgba", "depth"],
            pose_extractor_name="cylinder_pose_extractor",
            shuffle=False,
        )
        total_locations = len(extractor) / extractor.img_size[1]
        num_locations = min(total_locations, location_count)
        location_indices = np.round(
            np.linspace(0, total_locations, num_locations, endpoint=False)
        ).astype("int")
        for loc_i, pano_i in enumerate(location_indices):
            location_path = os.path.join(output_path, scene_name, f"location_{loc_i}")
            os.makedirs(location_path)
            cylinder_pano = extractor.create_panorama(pano_i)
            depth = cylinder_pano["depth"]
            imwrite(
                os.path.join(location_path, "actual_depth.png"), depth.astype("uint8")
            )
            imwrite(
                os.path.join(location_path, "scene.jpeg"),
                cylinder_pano["rgba"].astype("uint8")[..., :3],
            )

            disparity_map, layers = generate_mpi(
                os.path.join(location_path, "scene.jpeg")
            )
            imwrite(os.path.join(location_path, "predicted_depth.png"), disparity_map)
            os.makedirs(os.path.join(location_path, "layers"))
            for i, layer in enumerate(layers):
                imageio.imsave(
                    os.path.join(location_path, "layers", f"layer_{i}.png"),
                    (layer.numpy() * 255).astype("uint8"),
                )
            os.makedirs(os.path.join(location_path, "snapshots"))
            os.makedirs(os.path.join(location_path, "poses"))
            for i in range(snapshot_count):
                snapshot, pose = extractor.random_snapshot(
                    pano_i, offset_range=[-0.5, 0.5]
                )
                with open(
                    os.path.join(location_path, "poses", f"pose_{i}.json"), "w"
                ) as f:
                    json.dump(pose, f)
                imageio.imsave(
                    os.path.join(location_path, "snapshots", f"snapshot_{i}.png"),
                    snapshot,
                )

        extractor.close()


def render_mci_snapshots(data_path):
    """
    Generates MCI renders from the habitat-sim poses.
    - `data_path`: directory holding the habitat-sim renders.
    """
    locations = [
        os.path.split(path)[0]
        for path in glob.glob(os.path.join(data_path, "*", "*", "actual_depth.png"))
    ]
    for location in locations:
        # Calculate sigma
        predicted_depth = imread(os.path.join(location, "predicted_depth.png")).astype(
            "float32"
        )
        # interpolate depths that are far too close together
        predicted_depth = (
            (predicted_depth - predicted_depth.min())
            / (predicted_depth.max() - predicted_depth.min())
            * 255
        )

        actual_depth = imread(os.path.join(location, "actual_depth.png")).astype(
            "float32"
        )
        sigma = compute_sigma(predicted_depth, actual_depth)

        # Apply all coordinate transformations that align habitat to MCI to the `transform` matrix
        transform = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        poses = []
        pose_paths = sorted(
            glob.glob(os.path.join(location, "poses", "*.json"))
        )  # TODO: sort on pose number instead of string (will fail for pose count > 10)
        for path in pose_paths:
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
        out_dir = os.path.join(location, "predicted-snapshots")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        mci_renders = (
            render_image(
                (512, 512),
                os.path.join(location, "layers", "layer_%d.png"),
                eye,
                target,
                up=up,
                sigma=sigma,
            )
            for eye, target, up in poses
        )
        for i, render in enumerate(mci_renders):
            imageio.imsave(os.path.join(out_dir, f"snapshot_{i}.png"), render)


if __name__ == "__main__":
    main()
