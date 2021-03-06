"""
DEPRECATED
Image extractor which uses the equirectangular sensor to extract a spherical panorama.
Abandoned due to issues with sphere -> cylinder conversion.
"""

import habitat_sim
import numpy as np
from habitat_sim import bindings as hsim
from habitat_sim import registry as registry
import matplotlib.pyplot as plt
from habitat_sim.utils.data import ImageExtractor
import quaternion as qt
import json
import os


class EquirectangularExtractor(ImageExtractor):
    def _config_sim(self, scene_filepath, img_size):
        settings = {
            "width": img_size[1],  # Spatial resolution of the observations
            "height": img_size[0],
            "scene": scene_filepath,  # Scene path
            "default_agent": 0,
            "sensor_height": 1.5,  # Height of sensors in meters
            "color_sensor": True,  # RGBA sensor
            "depth_sensor": True,  # Depth sensor
            "silent": True,
        }

        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensor_specs = []
        if settings["color_sensor"]:
            color_sensor_spec = habitat_sim.sensor.EquirectangularSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [settings["height"], settings["width"]]
            color_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            sensor_specs.append(color_sensor_spec)

            color_cam = habitat_sim.sensor.CameraSensorSpec()
            color_cam.uuid = "color_snapshot"
            color_cam.sensor_type = habitat_sim.SensorType.COLOR
            color_cam.resolution = [settings["height"], settings["height"]]
            color_cam.postition = [0.0, settings["sensor_height"], 0.0]
            sensor_specs.append(color_cam)

        if settings["depth_sensor"]:
            depth_sensor_spec = habitat_sim.sensor.EquirectangularSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [settings["height"], settings["width"]]
            depth_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            sensor_specs.append(depth_sensor_spec)

        # create agent specifications
        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def positions(self, idx: range) -> tuple:
        poses = self.mode_to_data[self.mode.lower()]
        return [info[0] for info in poses[idx]]

    def random_snapshot(self, index):
        """
        Generates an image with a random position and rotation offset from the indicated panorama.
        - index: The index of the panorama in the main image
        returns the image and json describing the offset
        """
        cam_offset = np.random.uniform(-1, 1, 3)
        target = cam_offset + np.random.uniform(-1, 1, 3)
        return (
            self.create_snapshot(index, target, cam_offset),
            {"eye": list(cam_offset), "target": list(target), "up": [0, 1, 0]},
        )

    def create_snapshot(self, index, target, cam_offset=np.zeros(3)):
        """
            Generates an square snapshot based on look at coordinates relative to the
            center of the panorama at `index`.
            `index`: index of panorama corresponding to EquirectangularExtractor()[index].
            `target`: position that the camera looks at.
            `cam_offset`: camera offset from the center of the panorama.
        """

        poses = self.mode_to_data[self.mode.lower()]
        pos, _, fp = poses[index]

        # Only switch scene if it is different from the last one accessed
        if fp != self.cur_fp:
            self.sim.reconfigure(self._config_sim(fp, self.img_size))
            self.cur_fp = fp

        new_state = habitat_sim.AgentState()

        point = pos + cam_offset
        lap = pos + target

        new_state.position = point

        up = np.array([0.0, 1.0, 0.0])
        mat = lookAt(np.array([0.0, 0.0, 0.0]), lap - point, up)
        new_state.rotation = qt.from_rotation_matrix(mat[:3, :3])
        self.sim.agents[0].set_state(new_state)
        obs = self.sim.get_sensor_observations()
        return obs["color_snapshot"]


def lookAt(eye, center, up):
    F = center - eye

    f = normalize(F)
    if abs(f[1]) > 0.99:
        f = normalize(up) * np.sign(f[1])
        u = np.array([0, 0, 1])
        s = np.cross(f, u)
    else:
        s = np.cross(f, normalize(up))
        u = np.cross(normalize(s), f)
    M = np.eye(4)
    M[0:3, 0] = s
    M[0:3, 1] = u
    M[0:3, 2] = -f

    T = np.eye(4)
    T[3, 0:3] = -eye
    return M @ T


def normalize(vec):
    return vec / np.linalg.norm(vec)


if __name__ == "__main__":
    scene_filepath = "scenes/skokloster-castle.glb"
    extractor = EquirectangularExtractor(
        ["scenes/skokloster-castle.glb", "scenes/apartment_1.glb"],
        img_size=(512, 1024),
        output=["rgba", "depth"],
        shuffle=False,
    )
    os.makedirs("tmp", exist_ok=True)
    for i, obs in enumerate(extractor[:6]):
        plt.imsave(f"./tmp/pano_{i}.png", obs["rgba"])
        plt.imsave(f"./tmp/depth_{i}.png", obs["depth"], cmap="Greys")

    for i in range(6):
        img, pose = extractor.random_snapshot(0)
        plt.imsave(f"./snap/snapshot_{i}.png", img)
        with open(f"./snap/pose_{i}.json", "w") as out:
            json.dump(pose, out)

    extractor.close()
