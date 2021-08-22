
from panorama_extractor import PanoExtractor
import matplotlib.pyplot as plt
import json
from convert import Converter
import numpy as np
import imageio
import os
if __name__ == "__main__":
    scene_filepath = "scenes/skokloster-castle.glb"
    img_size = (512, 1024)
    extractor = PanoExtractor(
        scene_filepath,
        img_size=img_size,
        output=["rgba", "depth"],
        shuffle=True)
    cylinder_converter = Converter(*img_size)
    snap = extractor.random_snapshot(0)
    for i, obs in enumerate(extractor[:6]):
        plt.imsave(f"./sphere-panos/pano_{i}.png", obs["rgba"])
        depth = obs["depth"]
        depth[depth < 1.0] = 1.0
        depth = 255.0 / depth
        depth = depth.astype("uint8")
        depth = np.stack((depth,) * 3, axis=-1)

        imageio.imsave(f"./sphere-panos/depth_{i}.png",
                       depth)

    if not os.path.exists("test"):
        os.mkdir("test")
    for i in range(6):
        img, pose = extractor.random_snapshot(0)
        imageio.imsave(f"./test/snapshot_{i}.png", img)
        with open(f"./test/snapshot_{i}.json", "w") as out:
            json.dump(pose, out)

    extractor.close()
# python convert.py --input sphere-panos --output cylinder-panos
