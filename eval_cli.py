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
from mci_renderer import MCIRenderer, compute_sigma
from mesh_render.render_mesh import render_mesh
import imageio
import json
import shutil
from utils import check_dependencies
import importlib
from argparse import ArgumentParser

generate_mpi = None

PANO_DIMENSIONS = (512, 2048)
SNAPSHOT_DIMENSIONS = (PANO_DIMENSIONS[0],) * 2


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
        render_predicted_snapshots(args.data_dir)
    else:
        # Run evaluation of provided data
        render_predicted_snapshots(args.data_dir)


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
    render_predicted_snapshots(a["output_path"])


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

    render_predicted_snapshots(a["data_path"])


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
    ) + glob.glob(os.path.join(scene_path, "**", "*.ply"), recursive=True)
    if len(scene_paths) == 0:
        print(f"no scenes found in {scene_path}")
    for scene in scene_paths:
        scene_name = os.path.split(scene)[1].split(".")[0]
        os.makedirs(os.path.join(output_path, scene_name))
        extractor = CylinderExtractor(
            scene,
            img_size=PANO_DIMENSIONS,
            output=["rgba", "depth"],
            pose_extractor_name="cylinder_pose_extractor",
            shuffle=False,
        )
        total_locations = int(len(extractor) / extractor.img_size[1])
        num_locations = min(total_locations, location_count)
        location_indices = np.round(
            np.linspace(0, total_locations, num_locations, endpoint=False)
        ).astype("int")
        for loc_i, pano_i in enumerate(location_indices):
            location_path = os.path.join(output_path, scene_name, f"location_{loc_i}")
            os.makedirs(location_path)
            cylinder_pano = extractor.create_panorama(pano_i)
            depth = cylinder_pano["disparity"]
            np.save(os.path.join(location_path, "actual_disparity.npy"), depth)
            imwrite(
                os.path.join(location_path, "scene.jpeg"),
                cylinder_pano["rgba"].astype("uint8")[..., :3],
            )

            disparity_map, layers = generate_mpi(
                os.path.join(location_path, "scene.jpeg")
            )
            np.save(
                os.path.join(location_path, "predicted_disparity.npy"), disparity_map
            )
            layers = (np.array(layers) * 255.0).astype("uint8")
            os.makedirs(os.path.join(location_path, "layers"))
            for i, layer in enumerate(layers):
                imageio.imsave(
                    os.path.join(location_path, "layers", f"layer_{i}.png"), layer
                )
            os.makedirs(os.path.join(location_path, "snapshots"))
            os.makedirs(os.path.join(location_path, "poses"))
            for i in range(snapshot_count):
                snapshot, pose = extractor.random_snapshot(pano_i, offset=0.1)
                with open(
                    os.path.join(location_path, "poses", f"pose_{i}.json"), "w"
                ) as f:
                    json.dump(pose, f)
                imageio.imsave(
                    os.path.join(location_path, "snapshots", f"snapshot_{i}.png"),
                    snapshot,
                )

        extractor.close()


def render_predicted_snapshots(data_path):
    """
    Generates predition renders from the habitat-sim poses.
    - `data_path`: directory holding the habitat-sim renders.
    """
    locations = (
        os.path.split(path)[0]
        for path in glob.iglob(
            os.path.join(data_path, "*", "*", "actual_disparity.npy")
        )
    )
    for location in locations:
        print(f"Rendering poses for {location}")
        # Calculate sigma
        predicted_disparity = np.load(os.path.join(location, "predicted_disparity.npy"))
        # interpolate depths that are far too close together

        actual_disparity = np.load(os.path.join(location, "actual_disparity.npy"))
        sigma = compute_sigma(predicted_disparity, actual_disparity)
        # Apply all coordinate transformations that align habitat to MCI to the `transform` matrix
        transform = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        poses = []

        pose_paths = sorted(
            glob.glob(os.path.join(location, "poses", "*.json")),
            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
        )
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
        mci_dir = os.path.join(location, "mci-snapshots")
        if os.path.exists(mci_dir):
            shutil.rmtree(mci_dir)
        os.makedirs(mci_dir)
        mci_renderer = MCIRenderer(
            SNAPSHOT_DIMENSIONS,
            os.path.join(location, "layers", "layer_%d.png"),
            sigma=sigma,
        )
        mci_renders = (
            mci_renderer.render_image(eye, target, up=up) for eye, target, up in poses
        )
        for i, render in enumerate(mci_renders):
            imageio.imsave(os.path.join(mci_dir, f"snapshot_{i}.png"), render)
        del mci_renderer

        # Create mesh renders
        mesh_dir = os.path.join(location, "mesh-snapshots")
        if os.path.exists(mesh_dir):
            shutil.rmtree(mesh_dir)
        os.makedirs(mesh_dir)
        mesh_renders = (
            render_mesh(
                SNAPSHOT_DIMENSIONS,
                os.path.join(location, "scene.jpeg"),
                predicted_disparity,
                eye,
                target,
                up=up,
                sigma=sigma,
            )
            for eye, target, up in poses
        )
        for i, render in enumerate(mesh_renders):
            imageio.imsave(os.path.join(mesh_dir, f"snapshot_{i}.png"), render)


if __name__ == "__main__":
    main()
