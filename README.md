# Panoramic Synthesis Evaluation

Evaluation pipeline for [PanoSynthVR](https://dl.acm.org/doi/fullHtml/10.1145/3450618.3469144)

## Getting Started

Create an environment and install dependencies with [`conda`](https://docs.conda.io/en/latest/miniconda.html)

```bash
conda env create --file environment.yml
pre-commit install
```

Generate evaluation data using the CLI

```bash
python eval_cli.py
```

Or run the script using command line arguments

```bash
python eval_cli.py --scene-dir habitat/scenes --data-dir data --num-locations 3 --num-snapshots 1
```

| Argument      | Description                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| scene-dir     | Directory that holds .glb and .navmesh files for each scene                                                   |
| data-dir      | Data storage directory. Serves as the output of the habitat generation and the source of info for MCI renders |
| num-locations | The number of panorama locations sampled from each scene                                                      |
| num-snapshots | The number of snapshot render comparisons for each panorama location                                          |

## Pipeline Output

The pipeline outputs all data into the `/data` folder.

```
data
└── apartment_1
    └── location_0
        ├── actual_depth.png
        ├── layers
        │   ├── layer_0.png
        │   └── layer_31.png
        ├── poses
        │   └── pose_0.json
        ├── predicted_depth.png
        ├── predicted_snapshots
        │   └── snapshot_0.png
        ├── scene.jpeg
        └── snapshots
            └── snapshot_0.png
```

### Key Points

- The data folder contains different scenes, with names corresponding to the `.glb` files stored in `scene-dir`
- Each scene contains a list of locations based off of `num-locations`
- Locations contain panoramic images of the actual depth, predicted depth and scene, where the depths are disparity maps of the location
- Snapshots are square images with a random offset from the center of the panorama and a randomized look at point. Every snapshot render from habitat has a corresponding predicted-snapshot render from the MCI. Both renders use the information in the poses folder to correctly position the camera.

## Resources

- [Habitat Sim](https://github.com/facebookresearch/habitat-sim)
- [Matterport dataset](https://github.com/facebookresearch/habitat-sim/blob/master/DATASETS.md)
- [Replica dataset](https://github.com/facebookresearch/Replica-Dataset)
- [Parent Repository](https://github.com/richagadgil/PanoViewSynthesis)
