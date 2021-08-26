from typing import List, Tuple
import habitat_sim
import numpy as np
from numpy import float32, ndarray

from habitat_sim import bindings as hsim
from habitat_sim import registry as registry
import matplotlib.pyplot as plt
from habitat_sim.utils.data import ImageExtractor, PoseExtractor
import quaternion as qt
import json


class CylinderExtractor(ImageExtractor):
    def _config_sim(self, scene_filepath, img_size):
        settings = {
            "width": img_size[0] * 4,  # Ensure that the cylinder is the right resolution.
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
            color_sensor_spec = habitat_sim.sensor.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [
                settings["height"], 1]
            color_sensor_spec.hfov = 360.0 / settings["width"]
            color_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            sensor_specs.append(color_sensor_spec)


            color_cam = habitat_sim.sensor.CameraSensorSpec()
            color_cam.uuid = "color_snapshot"
            color_cam.sensor_type = habitat_sim.SensorType.COLOR
            color_cam.resolution = [
                settings["height"], settings["height"]]
            color_cam.postition = [0.0, settings["sensor_height"], 0.0]
            sensor_specs.append(color_cam)

        if settings["depth_sensor"]:
            depth_sensor_spec = habitat_sim.sensor.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [
                settings["height"], 1]
            depth_sensor_spec.hfov = 360.0 / settings["width"]
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
        return self.create_snapshot(index, target, cam_offset), {"eye": list(cam_offset), "target": list(target), "up": [0,1,0]}

    def create_snapshot(self, index, target, cam_offset=np.zeros(3)):
        """
            Generates an square snapshot based on look at coordinates relative to the 
            center of the panorama at `index`.
            `index`: index of panorama corresponding to PanoExtractor()[index].
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
        mat = lookAt(np.array([0.0, 0.0, 0.0]), lap -
                     point, up)
        new_state.rotation = qt.from_rotation_matrix(mat[:3, :3])
        self.sim.agents[0].set_state(new_state)
        obs = self.sim.get_sensor_observations()
        return obs["color_snapshot"]

    def create_panorama(self,index:int):
        img_size = 2048
        depth =[]
        color = []
        for img in self[img_size*index:(index+1)*img_size]:
            depth.append(img["depth"])
            color.append(img["rgba"])

        depth = np.concatenate(depth, axis=1)
        color = np.concatenate(color, axis=1)
        return {"depth": depth, "rgba": color}



# Pose extractor code
@registry.register_pose_extractor(name="cylinder_pose_extractor")
class CylinderPoseExtractor(PoseExtractor):
    def __init__(self,
                 topdown_views: List[Tuple[str, str, Tuple[float32, float32, float32]]],
                 meters_per_pixel: float = 0.1
                 ) -> None:
        super().__init__(topdown_views, meters_per_pixel)

    def extract_poses(
        self, view: ndarray, fp: str
    ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]]:
        # Determine the physical spacing between each camera position
        height, width = view.shape
        # We can modify this to be user-defined later
        dist = min(height, width) // 10
        cam_height = 3
        # Create a grid of camera positions
        n_gridpoints_width, n_gridpoints_height = (
            width // dist - 1,
            height // dist - 1,
        )
        # groups of xz points sampled from accessible areas in the scene
        gridpoints = []
        # Scene reachability mask with bounds away from walls.
        for h in range(n_gridpoints_height):
            for w in range(n_gridpoints_width):
                point = (dist + h * dist, dist + w * dist)
                if self._valid_point(*point, view):
                    gridpoints.append(point)
        # Generate a pose for each side of the cubemap
        poses = []
        for row, col in gridpoints:
            position = (col, cam_height, row)
            points_of_interest = self._panorama_extraction(
                position, view, dist)
            poses.extend([(position, poi, fp) for poi in points_of_interest])
        # Returns poses in 3D cartesian coordinate system
        return poses

    def _convert_to_scene_coordinate_system(
        self,
        poses: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]],
        ref_point: Tuple[float32, float32, float32],
    ) -> List[Tuple[Tuple[int, int], qt.quaternion, str]]:
        # Convert from topdown map coordinate system to that of the scene
        start_point = np.array(ref_point)
        converted_poses = []
        for i, pose in enumerate(poses):
            pos, look_at_point, filepath = pose

            new_pos = start_point + np.array(pos) * self.meters_per_pixel
            new_lap = start_point + \
                np.array(look_at_point) * self.meters_per_pixel
            displacement = new_lap - new_pos

            rot = qt.from_rotation_matrix(lookAt(np.array([0, 0, 0]),
                                                 displacement, np.array([0, 1, 0]))[:3,:3])
            converted_poses.append(
                (new_pos, rot, filepath))

        return converted_poses

    def _panorama_extraction(
        self, point: Tuple[int, int, int], view: ndarray, dist: int
    ) -> List[Tuple[int, int]]:
        neighbors = []
        for angle in np.linspace(0, np.pi * 2, 2048, endpoint=False):
            lap = np.array([np.sin(angle)* 3, 0, np.cos(angle) * 3]) + point
            neighbors.append(lap.tolist())
        return neighbors

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
    return M@T


def normalize(vec):
    return vec / np.linalg.norm(vec)


if __name__ == "__main__":
    scene_filepath = "habitat/scenes/skokloster-castle.glb"
    extractor = CylinderExtractor(
        scene_filepath,
        img_size=(512, 2048),
        output=["rgba", "depth"],
        pose_extractor_name="cylinder_pose_extractor",
        shuffle=False)
    index = 2
    img = extractor.create_panorama(index)
    plt.imsave("test.png", img["rgba"])

    extractor.close()


