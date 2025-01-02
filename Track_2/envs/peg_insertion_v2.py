import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

import copy
import json
import math
import time
from typing import Tuple, Union

import fcl
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import transforms3d as t3d
import warp as wp
from gymnasium import spaces
from path import Path
import sapien
import sapien.sensor as sapien_sensor
from sapien.utils.viewer import Viewer as viewer
from sapienipc.ipc_system import IPCSystem, IPCSystemConfig
from sapienipc.ipc_utils.user_utils import ipc_update_render_all

from envs.common_params import CommonParams
from envs.tactile_sensor_sapienipc import (
    TactileSensorSapienIPC,
    VisionTactileSensorSapienIPC,
)
from utils.common import randomize_params, suppress_stdout_stderr, get_time
from utils.geometry import quat_product
from utils.gym_env_utils import convert_observation_to_space
from utils.sapienipc_utils import build_sapien_entity_ABD

from loguru import logger as log
from utils.mem_monitor import *
import open3d as o3d
import cv2 as cv

wp.init()
wp_device = wp.get_preferred_device()


def evaluate_error_v2(info):
    """
    Evaluates the error between the peg and the hole based on the ground truth offset.

    This function calculates the L2 norm (Euclidean distance) of the ground truth offset,
    which represents the difference between the peg's current position and the target position.

    Parameters:
    - info (dict): A dictionary containing the environment's state information,
                   including the ground truth offset ('gt_offset_mm_deg') in millimeters and degrees.

    Returns:
    - error (float): The L2 norm of the ground truth offset, representing the error.

    """
    return np.linalg.norm(info["gt_offset_mm_deg"], ord=2)


class PegInsertionParams(CommonParams):
    """
    Stores parameters specific to the Peg Insertion environment.

    This class extends CommonParams and adds parameters that are specific to the peg insertion task.
    It includes parameters for gripper offsets, indentation depth, and friction coefficients for the peg and hole.

    Attributes:
    - gripper_x_offset_mm (float): The x-axis offset of the gripper in millimeters.
    - gripper_z_offset_mm (float): The z-axis offset of the gripper in millimeters.
    - indentation_depth_mm (float): The depth of gripper indentation in millimeters.
    - peg_friction (float): The friction coefficient of the peg.
    - hole_friction (float): The friction coefficient of the hole.

    """

    def __init__(
        self,
        gripper_x_offset_mm: float = 0.0,
        gripper_z_offset_mm: float = 0.0,
        indentation_depth_mm: float = 1.0,
        peg_friction: float = 1.0,
        hole_friction: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gripper_x_offset_mm = gripper_x_offset_mm
        self.gripper_z_offset_mm = gripper_z_offset_mm
        self.indentation_depth_mm = indentation_depth_mm
        self.peg_friction = peg_friction
        self.hole_friction = hole_friction


class PegInsertionSimEnvV2(gym.Env):
    """
    A simulation environment for a peg insertion task using the Gym interface.

    This class simulates the process of inserting a peg into a hole, providing a realistic
    physical interaction using the Sapien physics engine and IPC system.
    """

    def __init__(
        self,
        # reward
        step_penalty: float = 1.0,
        final_reward: float = 10.0,
        # peg file
        peg_hole_path_file: str = "",
        # peg position
        peg_x_max_offset_mm: float = 5.0,
        peg_y_max_offset_mm: float = 5.0,
        peg_theta_max_offset_deg: float = 10.0,
        peg_dist_z_mm: float = 10.0,
        peg_dist_z_diff_mm: float = 5.0,
        # peg movement
        max_action_mm_deg: np.ndarray = [1.0, 1.0, 1.0, 1.0],
        max_steps: int = 50,
        insertion_depth_mm: float = 1.0,
        # env randomization
        params=None,
        params_upper_bound=None,
        # device
        device: str = "cuda:0",
        # render
        no_render: bool = False,
        # for logging
        log_path=None,
        logger=None,
        env_type: str = "train",
        gui: bool = False,
        # for vision
        vision_params: dict = None,
        **kwargs,
    ):
        """
        Initialize the PegInsertionSimEnvV2.
        """
        # for logging
        time.sleep(np.random.rand(1)[0])
        if logger is None:
            self.logger = log
            self.logger.remove()
        else:
            self.logger = logger

        self.log_time = get_time()
        self.pid = os.getpid()
        if log_path is None:
            self.log_folder = track_path + "/envs/" + self.log_time
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)
        elif os.path.isdir(log_path):
            self.log_folder = log_path
        else:
            self.log_folder = Path(log_path).dirname()
        self.log_path = Path(
            os.path.join(
                self.log_folder,
                f"{self.log_time}_{env_type}_{self.pid}_PegInsertionEnvV2.log",
            )
        )
        print(self.log_path)
        self.logger.add(
            self.log_path,
            filter=lambda record: record["extra"]["name"] == self.log_time,
        )
        self.unique_logger = self.logger.bind(name=self.log_time)
        self.env_type = env_type
        self.gui = gui
        super(PegInsertionSimEnvV2, self).__init__()

        # Initialize environment parameters
        self.step_penalty = step_penalty
        self.final_reward = final_reward
        self.max_steps = max_steps
        peg_hole_path_file = Path(track_path) / peg_hole_path_file
        self.peg_hole_path_list = []
        with open(peg_hole_path_file, "r") as f:
            for l in f.readlines():
                self.peg_hole_path_list.append(
                    [ss.strip() for ss in l.strip().split(",")]
                )
        self.peg_x_max_offset_mm = peg_x_max_offset_mm
        self.peg_y_max_offset_mm = peg_y_max_offset_mm
        self.peg_theta_max_offset_deg = peg_theta_max_offset_deg
        self.peg_dist_z_mm = peg_dist_z_mm
        self.peg_dist_z_diff_mm = peg_dist_z_diff_mm
        self.insertion_depth_mm = insertion_depth_mm
        self.no_render = no_render

        if params is None:
            self.params_lb = PegInsertionParams()
        else:
            self.params_lb = copy.deepcopy(params)

        if params_upper_bound is None:
            self.params_ub = copy.deepcopy(self.params_lb)
        else:
            self.params_ub = copy.deepcopy(params_upper_bound)

        self.params = randomize_params(
            self.params_lb, self.params_ub
        )  # type: PegInsertionParams

        self.current_episode_elapsed_steps = 0
        self.sensor_grasp_center_init_mm_deg = np.array([0, 0, 0, 0])
        self.sensor_grasp_center_current_mm_deg = self.sensor_grasp_center_init_mm_deg

        # create IPC system
        ipc_system_config = IPCSystemConfig()
        # memory config
        ipc_system_config.max_scenes = 1
        ipc_system_config.max_surface_primitives_per_scene = 1 << 11
        ipc_system_config.max_blocks = 100000
        # scene config
        ipc_system_config.time_step = self.params.sim_time_step
        ipc_system_config.gravity = wp.vec3(0, 0, 0)
        ipc_system_config.d_hat = self.params.sim_d_hat  # 2e-4
        ipc_system_config.eps_d = self.params.sim_eps_d  # 1e-3
        ipc_system_config.eps_v = self.params.sim_eps_v  # 1e-2
        ipc_system_config.v_max = 1e-1
        ipc_system_config.kappa = self.params.sim_kappa  # 1e3
        ipc_system_config.kappa_affine = self.params.sim_kappa_affine
        ipc_system_config.kappa_con = self.params.sim_kappa_con
        ipc_system_config.ccd_slackness = self.params.ccd_slackness
        ipc_system_config.ccd_thickness = self.params.ccd_thickness
        ipc_system_config.ccd_tet_inversion_thres = self.params.ccd_tet_inversion_thres
        ipc_system_config.ee_classify_thres = self.params.ee_classify_thres
        ipc_system_config.ee_mollifier_thres = self.params.ee_mollifier_thres
        ipc_system_config.allow_self_collision = bool(self.params.allow_self_collision)
        # solver config
        ipc_system_config.newton_max_iters = int(
            self.params.sim_solver_newton_max_iters
        )  # key param
        ipc_system_config.cg_max_iters = int(self.params.sim_solver_cg_max_iters)
        ipc_system_config.line_search_max_iters = int(self.params.line_search_max_iters)
        ipc_system_config.ccd_max_iters = int(self.params.ccd_max_iters)
        ipc_system_config.precondition = "jacobi"
        ipc_system_config.cg_error_tolerance = self.params.sim_solver_cg_error_tolerance
        ipc_system_config.cg_error_frequency = int(
            self.params.sim_solver_cg_error_frequency
        )
        ipc_system_config.device = wp.get_device(device)
        self.unique_logger.info("device : " + str(ipc_system_config.device))
        self.ipc_system = IPCSystem(ipc_system_config)

        # build scene
        sapien_system = [
            sapien.physx.PhysxCpuSystem(),
            sapien.render.RenderSystem(device=device),
            self.ipc_system,
        ]
        if not no_render:
            self.scene = sapien.Scene(systems=sapien_system)
            self.scene.set_ambient_light([1.0, 1.0, 1.0])
            self.scene.add_directional_light([0, -1, -1], [1.0, 1.0, 1.0], True)
        else:
            self.scene = sapien.Scene(systems=sapien_system)

        self.viewer = None

        # add a camera to indicate shader
        if not no_render:
            cam_entity = sapien.Entity()
            cam = sapien.render.RenderCameraComponent(512, 512)
            cam_entity.add_component(cam)
            cam_entity.name = "camera"
            self.scene.add_entity(cam_entity)

        # vision info
        self.camera_size = (480, 640)
        self.vision_params = vision_params
        if self.vision_params["render_mode"] == "rt":
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_denoiser(
                self.vision_params["ray_tracing_denoiser"]
            )
            sapien.render.set_ray_tracing_samples_per_pixel(
                self.vision_params["ray_tracing_samples_per_pixel"]
            )
        if "point_cloud" in self.vision_params["vision_type"]:
            self.max_points = self.vision_params["max_points"]

        max_action_mm_deg = np.array(max_action_mm_deg)
        assert max_action_mm_deg.shape == (4,)
        self.max_action_mm_deg = max_action_mm_deg
        self.action_space = spaces.Box(
            low=-1, high=1, shape=max_action_mm_deg.shape, dtype=np.float32
        )
        self.default_observation = self.__get_sensor_default_observation__()
        self.observation_space = convert_observation_to_space(self.default_observation)

    def seed(self, seed=None):
        if seed is None:
            seed = (int(time.time() * 1000) % 10000 * os.getpid()) % 2**30
        np.random.seed(seed)

    def __get_sensor_default_observation__(self) -> dict:

        meta_file = self.params.tac_sensor_meta_file
        meta_file = Path(track_path) / "assets" / meta_file
        with open(meta_file, "r") as f:
            config = json.load(f)
        meta_dir = Path(meta_file).dirname()
        on_surface_np = np.loadtxt(meta_dir / config["on_surface"]).astype(np.int32)
        initial_surface_pts = np.zeros((np.sum(on_surface_np), 3)).astype(float)

        obs = {
            "gt_direction": np.zeros((1,), dtype=np.float32),
            "gt_offset": np.zeros((4,), dtype=np.float32),
            "relative_motion": np.zeros((4,), dtype=np.float32),
            "surface_pts": np.stack([np.stack([initial_surface_pts] * 2)] * 2),
        }

        if "rgb" in self.vision_params["vision_type"]:
            obs["rgb_picture"] = np.zeros(
                (self.camera_size[0], self.camera_size[1], 3), dtype=np.uint8
            )
        if "depth" in self.vision_params["vision_type"]:
            obs["depth_picture"] = np.zeros(
                (self.camera_size[0], self.camera_size[1]), dtype=np.float32
            )
        if "point_cloud" in self.vision_params["vision_type"]:
            obs["point_cloud"] = np.zeros(
                (self.camera_size[0], self.camera_size[1], 3), dtype=np.float32
            )
            # for refrerence method
            obs["object_point_cloud"] = np.zeros(
                (2, self.max_points, 3), dtype=np.float32
            )

        return obs

    def reset(self, offset_mm_deg=None, seed=None, options=None) -> Tuple[dict, dict]:

        self.unique_logger.info(
            "*************************************************************"
        )
        self.unique_logger.info(self.env_type + " reset once")
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.params = randomize_params(self.params_lb, self.params_ub)
        self.current_episode_elapsed_steps = 0

        if offset_mm_deg:
            self.unique_logger.info(f"given offset_mm_deg: {offset_mm_deg}")
            offset_mm_deg = np.array(offset_mm_deg).astype(float)

        offset_mm_deg = self._initialize(offset_mm_deg)
        self.unique_logger.info(f"after initialize offset_mm_deg: {offset_mm_deg}")

        self.init_left_surface_pts_m = self.no_contact_surface_mesh[0]
        self.init_right_surface_pts_m = self.no_contact_surface_mesh[1]
        self.init_offset_of_current_episode_mm_deg = offset_mm_deg
        self.current_offset_of_current_episode_mm_deg = offset_mm_deg
        info = self.get_info()
        self.error_evaluation_list = []
        self.error_evaluation_list.append(evaluate_error_v2(info))
        obs = self.get_obs(info)

        return obs, info

    def _initialize(self, offset_mm_deg: Union[np.ndarray, None]) -> np.ndarray:
        """
        offset_mm_deg: (x_offset in mm, y_offset in mm, theta_offset in degree, z_offset in mm,  choose_left)
        """

        # randomly choose peg and hole
        self.peg_index = np.random.randint(
            len(self.peg_hole_path_list)
        )  # only one peg and hole in Track 2
        peg_path_l, peg_path_r, hole_path = self.peg_hole_path_list[self.peg_index]
        y_start_mm = 45.25 / 2  # mm, based on assets/peg_insertion/hole_2.5mm.STL
        z_target_mm = 6.9  # mm, based on assets/peg_insertion/hole_2.5mm.STL
        self.z_target_mm = z_target_mm
        choose_left = np.random.random() < 0.5
        if offset_mm_deg is not None:
            if len(offset_mm_deg) > 4:
                choose_left = offset_mm_deg[4]
                offset_mm_deg = offset_mm_deg[:4]
        if choose_left:
            self.unique_logger.info("choose left")
            peg_path = peg_path_l
            self.y_target_mm = 45.25  # mm, based on assets/peg_insertion/sim_hole_base_2.0mm.STL
            self.gt_direction = np.ones((1,), dtype=np.float32)
        else:
            self.unique_logger.info("choose right")
            peg_path = peg_path_r
            self.y_target_mm = 0  # mm, based on assets/peg_insertion/sim_hole_base_2.0mm.STL
            self.gt_direction = -np.ones((1,), dtype=np.float32)

        # to avoid the first frame black
        while True:
            # remove all entities except camera
            for e in self.scene.entities:
                if "camera" not in e.name:
                    e.remove_from_scene()
            self.ipc_system.rebuild()

            # get peg and hole path
            asset_dir = Path(track_path) / "assets"
            peg_path = asset_dir / peg_path
            hole_path = asset_dir / hole_path

            # add hole to the sapien scene
            with suppress_stdout_stderr():
                self.hole_entity, hole_abd = build_sapien_entity_ABD(
                    hole_path,
                    density=500.0,
                    color=[0.0, 0.0, 1.0, 0.95],
                    friction=self.params.hole_friction,
                    no_render=self.no_render,
                )  # blue
            self.hole_ext = os.path.splitext(hole_path)[-1]
            self.hole_entity.set_name("hole")
            self.hole_abd = hole_abd
            if self.hole_ext == ".msh":
                self.hole_upper_z_m = hole_height_m = np.max(
                    hole_abd.tet_mesh.vertices[:, 2]
                ) - np.min(hole_abd.tet_mesh.vertices[:, 2])
            else:
                self.hole_upper_z_m = hole_height_m = np.max(
                    hole_abd.tri_mesh.vertices[:, 2]
                ) - np.min(hole_abd.tri_mesh.vertices[:, 2])
            self.scene.add_entity(self.hole_entity)

            # add peg model
            with suppress_stdout_stderr():
                self.peg_entity, peg_abd = build_sapien_entity_ABD(
                    peg_path,
                    density=500.0,
                    color=[1.0, 0.0, 0.0, 0.95],
                    friction=self.params.peg_friction,
                    no_render=self.no_render,
                )  # red
            self.peg_ext = os.path.splitext(peg_path)[-1]
            self.peg_abd = peg_abd
            self.peg_entity.set_name("peg")
            if self.peg_ext == ".msh":
                peg_width_m = np.max(peg_abd.tet_mesh.vertices[:, 1]) - np.min(
                    peg_abd.tet_mesh.vertices[:, 1]
                )
                peg_height_m = np.max(peg_abd.tet_mesh.vertices[:, 2]) - np.min(
                    peg_abd.tet_mesh.vertices[:, 2]
                )
                self.peg_bottom_pts_id = np.where(
                    peg_abd.tet_mesh.vertices[:, 2]
                    < np.min(peg_abd.tet_mesh.vertices[:, 2]) + 1e-4
                )[0]
            else:
                peg_width_m = np.max(peg_abd.tri_mesh.vertices[:, 1]) - np.min(
                    peg_abd.tri_mesh.vertices[:, 1]
                )
                peg_height_m = np.max(peg_abd.tri_mesh.vertices[:, 2]) - np.min(
                    peg_abd.tri_mesh.vertices[:, 2]
                )
                self.peg_bottom_pts_id = np.where(
                    peg_abd.tri_mesh.vertices[:, 2]
                    < np.min(peg_abd.tri_mesh.vertices[:, 2]) + 1e-4
                )[0]

            # generate random and valid offset
            if offset_mm_deg is None:
                x_offset_mm = (np.random.rand() * 2 - 1) * self.peg_x_max_offset_mm
                y_offset_mm = (np.random.rand() * 2 - 1) * self.peg_y_max_offset_mm
                theta_offset_deg = (
                    np.random.rand() * 2 - 1
                ) * self.peg_theta_max_offset_deg
                z_offset_mm = (
                    self.peg_dist_z_mm
                    + (np.random.rand() * 2 - 1) * self.peg_dist_z_diff_mm
                )
                # no need to check collision here.
                offset_mm_deg = np.array(
                    [x_offset_mm, y_offset_mm, theta_offset_deg, z_offset_mm]
                )
            else:
                x_offset_mm, y_offset_mm, theta_offset_deg, z_offset_mm = (
                    offset_mm_deg[0],
                    offset_mm_deg[1],
                    offset_mm_deg[2],
                    offset_mm_deg[3],
                )
            # add peg to the scene
            epsilon = 0.001
            init_pos_m = (
                x_offset_mm / 1000,
                y_start_mm / 1000 + y_offset_mm / 1000,
                hole_height_m
                + z_offset_mm / 1000
                + epsilon,  # move to the initial z-distance.
            )
            init_theta_offset_rad = theta_offset_deg * np.pi / 180
            peg_offset_quat = t3d.quaternions.axangle2quat(
                (0, 0, 1), init_theta_offset_rad, True
            )
            self.peg_entity.set_pose(sapien.Pose(p=init_pos_m, q=peg_offset_quat))
            self.scene.add_entity(self.peg_entity)

            # add tactile sensors to the sapien scene
            gripper_x_offset_m = self.params.gripper_x_offset_mm / 1000  # mm to m
            gripper_z_offset_m = self.params.gripper_z_offset_mm / 1000
            sensor_grasp_center_m = np.array(
                (
                    math.cos(init_theta_offset_rad) * gripper_x_offset_m
                    + init_pos_m[0],
                    math.sin(init_theta_offset_rad) * gripper_x_offset_m
                    + init_pos_m[1],
                    peg_height_m + init_pos_m[2] + gripper_z_offset_m,
                )
            )
            init_pos_l_m = (
                -math.sin(init_theta_offset_rad) * (peg_width_m / 2 + 0.0020 + 0.0001)
                + sensor_grasp_center_m[0],
                math.cos(init_theta_offset_rad) * (peg_width_m / 2 + 0.0020 + 0.0001)
                + sensor_grasp_center_m[1],
                sensor_grasp_center_m[2],
            )
            init_pos_r_m = (
                math.sin(init_theta_offset_rad) * (peg_width_m / 2 + 0.0020 + 0.0001)
                + sensor_grasp_center_m[0],
                -math.cos(init_theta_offset_rad) * (peg_width_m / 2 + 0.0020 + 0.0001)
                + sensor_grasp_center_m[1],
                sensor_grasp_center_m[2],
            )
            init_rot_l = quat_product(peg_offset_quat, (0.5, 0.5, 0.5, -0.5))
            init_rot_r = quat_product(peg_offset_quat, (0.5, -0.5, 0.5, 0.5))
            with suppress_stdout_stderr():
                self._add_tactile_sensors(
                    init_pos_l_m, init_rot_l, init_pos_r_m, init_rot_r
                )

            # get init sensor center
            sensor_grasp_center_mm = tuple(
                (x * 1000 + y * 1000) / 2 for x, y in zip(init_pos_l_m, init_pos_r_m)
            )
            self.sensor_grasp_center_init_mm_deg = np.array(
                sensor_grasp_center_mm[:2]
                + (init_theta_offset_rad / np.pi * 180,)
                + sensor_grasp_center_mm[2:]
            )
            self.sensor_grasp_center_current_mm_deg = (
                self.sensor_grasp_center_init_mm_deg.copy()
            )

            # add depth sensor
            sensor_config = sapien_sensor.StereoDepthSensorConfig()
            sensor_config.rgb_resolution = (self.camera_size[1], self.camera_size[0])
            sensor_config.ir_resolution = (self.camera_size[1], self.camera_size[0])
            ir_intrinsic_matrix = np.array(
                [
                    [595.8051147460938, 0.0, 315.040283203125],
                    [0.0, 595.8051147460938, 246.26866149902344],
                    [0.0, 0.0, 1.0],
                ]
            )
            rgb_intrinsic_matrix = np.array(
                [
                    [604.3074951171875, 0.0, 317.234130859375],
                    [0.0, 604.3074951171875, 233.79444885253906],
                    [0.0, 0.0, 1.0],
                ]
            )
            sensor_config.rgb_intrinsic = rgb_intrinsic_matrix
            sensor_config.ir_intrinsic = ir_intrinsic_matrix
            sensor_config.min_depth = 0.01
            sensor_config.max_depth = 1.0
            sensor_config.trans_pose_l = sapien.Pose([0, -0.01502214651554823, 0])
            sensor_config.trans_pose_r = sapien.Pose([0, -0.07004545629024506, 0])
            self.main_cam = Depth_sensor(sensor_config, sapien.Entity())
            self.main_cam.set_name("Depth_Cam")
            self.scene.add_entity(self.main_cam.get_entity())

            self.scene.step()
            self.main_cam.set_pose(
                self._gen_camera_pose(
                    y=(self.sensor_grasp_center_init_mm_deg[1] + self.y_target_mm)
                    / 1000,
                    z=(self.sensor_grasp_center_init_mm_deg[2] + self.z_target_mm)
                    / 1000,
                )
            )
            # update render
            # avoid the first frame black
            self.scene.update_render()
            ipc_update_render_all(self.scene)
            self.main_cam.reset()
            rgb_raw = self.main_cam.get_rgb_data()
            rgb = (rgb_raw * 255).astype(np.uint8)
            blue_mask = (rgb[:, :, 2] > 200) & (rgb[:, :, 1] < 100)
            red_mask = (
                (rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)
            )
            if np.any(blue_mask) and np.any(red_mask):
                break
            # break

        # gui initialization
        if self.gui:
            self.viewer = viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_pose(
                sapien.Pose(
                    [-0.0477654, 0.0621954, 0.086787],
                    [0.846142, 0.151231, 0.32333, -0.395766],
                )
            )
            self.viewer.window.set_camera_parameters(0.001, 10.0, np.pi / 2)
            pause = True
            while pause:
                if self.viewer.window.key_down("c"):
                    pause = False
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

        # grasp the peg
        grasp_step = max(
            round(
                (0.1 + self.params.indentation_depth_mm)
                / 1000
                / 5e-3
                / (self.params.sim_time_step if self.params.sim_time_step != 0 else 1)
            ),
            1,
        )
        grasp_speed = (
            (0.1 + self.params.indentation_depth_mm)
            / 1000
            / grasp_step
            / self.params.sim_time_step
        )
        for _ in range(grasp_step):
            self.tactile_sensor_1.set_active_v(
                [
                    grasp_speed * math.sin(init_theta_offset_rad),
                    -grasp_speed * math.cos(init_theta_offset_rad),
                    0,
                ]
            )
            self.tactile_sensor_2.set_active_v(
                [
                    -grasp_speed * math.sin(init_theta_offset_rad),
                    grasp_speed * math.cos(init_theta_offset_rad),
                    0,
                ]
            )
            with suppress_stdout_stderr():
                self.hole_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
                )  # hole stays static
                self.ipc_system.step()
                self.scene.step()
            self.tactile_sensor_1.step()
            self.tactile_sensor_2.step()
            self.scene.update_render()
            ipc_update_render_all(self.scene)
            if self.gui:
                self.viewer.render()

        if isinstance(
            self.tactile_sensor_1, VisionTactileSensorSapienIPC
        ) and isinstance(self.tactile_sensor_2, VisionTactileSensorSapienIPC):
            self.tactile_sensor_1.set_reference_surface_vertices_camera()
            self.tactile_sensor_2.set_reference_surface_vertices_camera()
        self.no_contact_surface_mesh = copy.deepcopy(
            self._get_sensor_surface_vertices()
        )

        offset_mm_deg[1] = offset_mm_deg[1] + y_start_mm - self.y_target_mm
        offset_mm_deg[3] = offset_mm_deg[3] + hole_height_m * 1000 - self.z_target_mm

        return offset_mm_deg

    def _add_tactile_sensors(self, init_pos_l, init_rot_l, init_pos_r, init_rot_r):

        self.tactile_sensor_1 = TactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_l,
            init_rot=init_rot_l,
            elastic_modulus=self.params.tac_elastic_modulus_l,
            poisson_ratio=self.params.tac_poisson_ratio_l,
            density=self.params.tac_density_l,
            friction=self.params.tac_friction,
            name="tactile_sensor_1",
            no_render=self.no_render,
            logger=self.unique_logger,
        )

        self.tactile_sensor_2 = TactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_r,
            init_rot=init_rot_r,
            elastic_modulus=self.params.tac_elastic_modulus_r,
            poisson_ratio=self.params.tac_poisson_ratio_r,
            density=self.params.tac_density_r,
            friction=self.params.tac_friction,
            name="tactile_sensor_2",
            no_render=self.no_render,
            logger=self.unique_logger,
        )

    def step(self, action) -> Tuple[dict, float, bool, bool, dict]:
        """
        Advances the simulation by one step using the provided action.

        This method is called to execute one step of the simulation, where the action is applied to the environment.
        It updates the environment state, calculates the reward, and checks for termination conditions.

        Parameters:
        - action (np.ndarray): A numpy array representing the action to be taken, scaled to millimeters and degrees.

        Returns:
        - obs (dict): The observation of the environment after applying the action.
        - reward (float): The reward earned for the action taken.
        - terminated (bool): Whether the episode has terminated.
        - truncated (bool): Whether the episode was truncated (e.g., due to exceeding maximum steps).
        - info (dict): Additional information about the environment's state.
        """

        self.current_episode_elapsed_steps += 1
        self.unique_logger.info(
            "#############################################################"
        )
        self.unique_logger.info(
            f"current_episode_elapsed_steps: {self.current_episode_elapsed_steps}"
        )
        self.unique_logger.info(f"action: {action}")
        action_mm_deg = np.array(action).flatten() * self.max_action_mm_deg
        self.unique_logger.info(f"action_mm_deg: {action_mm_deg}")
        self._sim_step(action_mm_deg)

        info = self.get_info()
        self.unique_logger.info(f"info: {info}")
        obs = self.get_obs(info)
        reward = self.get_reward(info)
        terminated = self.get_terminated(info)
        truncated = self.get_truncated(info)
        self.unique_logger.info(
            "#############################################################"
        )
        return obs, reward, terminated, truncated, info

    def _sim_step(self, action_mm_deg):
        """
        Executes a single simulation step with the given action in millimeters and degrees.

        This method applies the action to the peg by updating its position and orientation in the simulation environment.
        It handles the physical interaction between the peg and the hole, as well as the tactile sensors.

        Parameters:
        - action_mm_deg (np.ndarray): A numpy array representing the action to be taken, containing x and y displacements in millimeters and a rotation in degrees.

        Returns:
        - None

        """
        action_mm_deg = np.clip(
            action_mm_deg, -self.max_action_mm_deg, self.max_action_mm_deg
        )
        current_theta_rad = (
            self.current_offset_of_current_episode_mm_deg[2] * np.pi / 180
        )
        action_x_mm, action_y_mm = action_mm_deg[:2] @ [
            math.cos(current_theta_rad),
            -math.sin(current_theta_rad),
        ], action_mm_deg[:2] @ [
            math.sin(current_theta_rad),
            math.cos(current_theta_rad),
        ]
        action_theta_deg = action_mm_deg[2]
        action_z_mm = action_mm_deg[3]

        self.current_offset_of_current_episode_mm_deg[0] += action_x_mm
        self.current_offset_of_current_episode_mm_deg[1] += action_y_mm
        self.current_offset_of_current_episode_mm_deg[2] += action_theta_deg
        self.current_offset_of_current_episode_mm_deg[3] += action_z_mm
        self.sensor_grasp_center_current_mm_deg[0] += action_x_mm
        self.sensor_grasp_center_current_mm_deg[1] += action_y_mm
        self.sensor_grasp_center_current_mm_deg[2] += action_theta_deg
        self.sensor_grasp_center_current_mm_deg[3] += action_z_mm

        action_sim_mm_deg = np.array(
            [action_x_mm, action_y_mm, action_theta_deg, action_z_mm]
        )
        sensor_grasp_center_m = (
            self.tactile_sensor_1.current_pos + self.tactile_sensor_2.current_pos
        ) / 2

        x_m = action_sim_mm_deg[0] / 1000
        y_m = action_sim_mm_deg[1] / 1000
        theta_rad = action_sim_mm_deg[2] * np.pi / 180
        z_m = action_sim_mm_deg[3] / 1000

        action_substeps = max(
            1,
            round(
                (max(abs(x_m), abs(y_m), abs(z_m)) / 5e-3) / self.params.sim_time_step
            ),
        )
        action_substeps = max(
            action_substeps, round((abs(theta_rad) / 0.2) / self.params.sim_time_step)
        )
        v_x = x_m / self.params.sim_time_step / action_substeps
        v_y = y_m / self.params.sim_time_step / action_substeps
        v_theta = theta_rad / self.params.sim_time_step / action_substeps
        v_z = z_m / self.params.sim_time_step / action_substeps

        for _ in range(action_substeps):
            self.tactile_sensor_1.set_active_v_r(
                [v_x, v_y, v_z],
                sensor_grasp_center_m,
                (0, 0, 1),
                v_theta,
            )
            self.tactile_sensor_2.set_active_v_r(
                [v_x, v_y, v_z],
                sensor_grasp_center_m,
                (0, 0, 1),
                v_theta,
            )
            with suppress_stdout_stderr():
                self.hole_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
                )  # hole stays static
                self.ipc_system.step()
                self.scene.step()
            state1 = self.tactile_sensor_1.step()
            state2 = self.tactile_sensor_2.step()
            sensor_grasp_center_m = (
                self.tactile_sensor_1.current_pos + self.tactile_sensor_2.current_pos
            ) / 2
            if (not state1) or (not state2):
                self.error_too_large = True
            if self.gui:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

    def get_info(self) -> dict:
        """
        Retrieves additional information about the environment's state.

        This method collects various metrics and data points that describe the current state of the simulation,
        including the positions of the peg and hole, surface differences, and success criteria.

        Returns:
        - info (dict): A dictionary containing the environment's state information.
        """
        info = {"steps": self.current_episode_elapsed_steps}

        observation_left_surface_pts_m, observation_right_surface_pts_m = (
            self._get_sensor_surface_vertices()
        )
        l_diff_m = np.mean(
            np.linalg.norm(
                self.init_left_surface_pts_m - observation_left_surface_pts_m, axis=-1
            )
        )
        r_diff_m = np.mean(
            np.linalg.norm(
                self.init_right_surface_pts_m - observation_right_surface_pts_m, axis=-1
            )
        )
        info["surface_diff_m"] = np.array([l_diff_m, r_diff_m])
        info["tactile_movement_too_large"] = False
        if l_diff_m > 1.5e-3 or r_diff_m > 1.5e-3:
            info["tactile_movement_too_large"] = True

        info["too_many_steps"] = False
        if self.current_episode_elapsed_steps >= self.max_steps:
            info["too_many_steps"] = True

        info["error_too_large"] = False
        x_offset_mm, y_offset_mm, theta_offset_deg, z_offset_mm = (
            self.current_offset_of_current_episode_mm_deg
        )
        if (
            np.abs(x_offset_mm) > 12 + 1e-5
            or np.abs(y_offset_mm) > 30 + 1e-5
            or np.abs(theta_offset_deg) > 20 + 1e-5
            or np.abs(z_offset_mm - self.hole_upper_z_m * 1000) > 15 + 1e-5
        ):
            info["error_too_large"] = True

        peg_relative_z_mm = self._get_peg_relative_z_mm()
        info["peg_relative_z_mm"] = peg_relative_z_mm
        info["is_success"] = False
        if (
            not info["error_too_large"]
            and not info["too_many_steps"]
            and not info["tactile_movement_too_large"]
            and np.sum(peg_relative_z_mm < -self.insertion_depth_mm)
            == peg_relative_z_mm.shape[0]
            and np.abs(x_offset_mm) < 6.0
            and np.abs(y_offset_mm) < 6.0
            and np.abs(theta_offset_deg) < 10.0
        ):
            info["is_success"] = True

        info["relative_motion_mm_deg"] = (
            self.sensor_grasp_center_current_mm_deg
            - self.sensor_grasp_center_init_mm_deg
        )

        info["gt_offset_mm_deg"] = self.current_offset_of_current_episode_mm_deg

        return info

    def get_obs(self, info) -> dict:
        """
        Constructs the observation dictionary from the environment's state information.

        This method takes the state information and processes it to create an observation that can be used by an agent or algorithm.
        The observation includes the ground truth direction, ground truth offset, relative motion, surface points of the tactile sensors,
        and some vision information.

        Parameters:
        - info (dict): A dictionary containing the environment's state information, obtained from `get_info`.

        Returns:
        - obs (dict): A dictionary containing the observation data.
        """

        obs_dict = dict()

        obs_dict["gt_direction"] = self.gt_direction
        obs_dict["gt_offset"] = np.array(info["gt_offset_mm_deg"]).astype(np.float32)
        obs_dict["relative_motion"] = np.array(info["relative_motion_mm_deg"]).astype(
            np.float32
        )
        observation_left_surface_pts, observation_right_surface_pts = (
            self._get_sensor_surface_vertices()
        )
        obs_dict["surface_pts"] = (
            np.stack(
                [
                    np.stack(
                        [
                            self.init_left_surface_pts_m,
                            observation_left_surface_pts,
                        ]
                    ),
                    np.stack(
                        [
                            self.init_right_surface_pts_m,
                            observation_right_surface_pts,
                        ]
                    ),
                ]
            ).astype(np.float32),
        )

        # visual observation
        self.main_cam.set_pose(
            self._gen_camera_pose(
                y=(self.current_offset_of_current_episode_mm_deg[1] + self.y_target_mm)
                / 1000,
                z=(self.current_offset_of_current_episode_mm_deg[2] + self.z_target_mm)
                / 1000,
            )
        )

        # update render
        self.scene.update_render()
        ipc_update_render_all(self.scene)
        self.main_cam.reset()
        if "rgb" in self.vision_params["vision_type"]:
            rgb_raw = self.main_cam.get_rgb_data()
            rgb = (rgb_raw * 255).astype(np.uint8)
            obs_dict["rgb_picture"] = rgb
        if "depth" in self.vision_params["vision_type"]:
            depth = self.main_cam.get_depth_data()
            obs_dict["depth_picture"] = depth
        if "point_cloud" in self.vision_params["vision_type"]:
            points_frame = self.main_cam.get_point_cloud(
                self.vision_params["gt_point_cloud"]
            )
            obs_dict["point_cloud"] = points_frame
            # for reference method
            rgb_raw = self.main_cam.get_rgb_data()
            rgb = (rgb_raw * 255).astype(np.uint8)
            obs_dict["object_point_cloud"], point_list = self._parse_points(
                rgb, obs_dict["point_cloud"]
            )

        ## if you want to save the image and point cloud, uncomment the following code
        # folder_name = self.log_folder
        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name)
        # temp_name = (
        #     folder_name
        #     + f"/{self.current_episode_elapsed_steps}_"
        #     + self.vision_params["render_mode"]
        # )
        # if "rgb" in self.vision_params["vision_type"]:
        #     plt.figure(figsize=(12, 8))
        #     plt.subplot(111)
        #     plt.title("RGB Image")
        #     plt.imshow(obs_dict["rgb_picture"])
        #     plt.savefig(temp_name + "_rgb.png", dpi=500)
        #     plt.close()
        # if "depth" in self.vision_params["vision_type"]:
        #     plt.figure(figsize=(12, 8))
        #     plt.subplot(111)
        #     plt.title("Depth Map")
        #     plt.imshow(obs_dict["depth_picture"])
        #     plt.savefig(temp_name + "_depth.png", dpi=500)
        #     plt.close()
        # if "point_cloud" in self.vision_params["vision_type"]:
        #     point_world = point_list[0]
        #     point_peg = point_list[1]
        #     point_hole = point_list[2]
        #     pcd_all = o3d.geometry.PointCloud()
        #     pcd_all.points = o3d.utility.Vector3dVector(point_world)
        #     o3d.io.write_point_cloud(temp_name + "_all.pcd", pcd_all)
        #     pcd_peg = o3d.geometry.PointCloud()
        #     pcd_peg.points = o3d.utility.Vector3dVector(point_peg)
        #     o3d.io.write_point_cloud(temp_name + "_peg.pcd", pcd_peg)
        #     pcd_hole = o3d.geometry.PointCloud()
        #     pcd_hole.points = o3d.utility.Vector3dVector(point_hole)
        #     o3d.io.write_point_cloud(temp_name + "_hole.pcd", pcd_hole)

        return obs_dict

    def get_reward(self, info) -> float:
        """
        Calculates the reward based on the environment's state information.

        This method determines the reward for the current state by evaluating the progress towards solving the peg insertion task.
        The reward is based on the error evaluation, step penalty, and final reward criteria.

        Parameters:
        - info (dict): A dictionary containing the environment's state information, obtained from `get_info`.

        Returns:
        - reward (float): The calculated reward for the current state and action.
        """

        self.error_evaluation_list.append(evaluate_error_v2(info))

        reward_part_1 = self.error_evaluation_list[-2] - self.error_evaluation_list[-1]
        reward_part_2 = -self.step_penalty
        reward_part_3 = 0

        if info["error_too_large"] or info["tactile_movement_too_large"]:
            reward_part_3 += (
                -2
                * self.step_penalty
                * (self.max_steps - self.current_episode_elapsed_steps)
                + self.step_penalty
            )
        elif info["is_success"]:
            reward_part_3 += self.final_reward

        reward = reward_part_1 + reward_part_2 + reward_part_3
        self.unique_logger.info(
            f"reward: {reward}, reward_part_1: {reward_part_1}, reward_part_2: {reward_part_2}, reward_part_3: {reward_part_3}"
        )

        return reward

    def get_truncated(self, info) -> bool:
        """
        Determines if the episode was truncated due to specific conditions.

        An episode can be truncated if certain conditions are met, such as exceeding the maximum
        number of steps or if the agent's actions lead to an undesirable state.

        Parameters:
        - info (dict): A dictionary containing the environment's state information, obtained from `get_info`.

        Returns:
        - truncated (bool): True if the episode was truncated, False otherwise.
        """
        return (
            info["too_many_steps"]
            or info["tactile_movement_too_large"]
            or info["error_too_large"]
        )

    def get_terminated(self, info) -> bool:
        """
        Determines if the episode has terminated due to success or failure.

        An episode can terminate either because the agent successfully opens the lock or because it fails to do so within the allowed steps.

        Parameters:
        - info (dict): A dictionary containing the environment's state information, obtained from `get_info`.

        Returns:
        - terminated (bool): True if the episode has terminated, False otherwise.
        """
        return info["is_success"]

    def _parse_points(self, rgb, points):

        white_mask = (rgb[:, :, 0] > 220) & (rgb[:, :, 1] > 220) & (rgb[:, :, 2] > 220)
        red_mask = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)
        blue_mask = (rgb[:, :, 2] > 200) & (rgb[:, :, 1] < 100)

        not_white_mask = np.logical_not(white_mask)
        mask0 = np.logical_and.reduce([red_mask, np.logical_not(blue_mask)])
        mask2 = np.logical_and.reduce([blue_mask, np.logical_not(red_mask)])

        object_all = points[not_white_mask]
        points_peg = points[mask0]
        points_hole = points[mask2]

        object_point_cloud = np.stack(
            [
                self._sample_points(points_peg, self.max_points),
                self._sample_points(points_hole, self.max_points),
            ]
        )

        return object_point_cloud, [object_all, points_peg, points_hole]

    def _get_sensor_surface_vertices(self) -> list[np.ndarray, np.ndarray]:
        """
        Retrieves the surface vertices of the tactile sensors in the world coordinate system.

        This function is used to get the current surface vertices of the tactile sensors, which can be used to
        calculate the relative motion or deformation of the surface.

        Returns:
        - vertices (list of np.ndarray): A list containing the surface vertices of the left and right tactile sensors.

        """
        return [
            self.tactile_sensor_1.get_surface_vertices_sensor(),
            self.tactile_sensor_2.get_surface_vertices_sensor(),
        ]

    def _get_peg_relative_z_mm(self) -> np.ndarray:
        """
        Calculates the relative z-coordinates of the peg's bottom points with respect to the hole's upper surface.

        This method is used to determine how far the peg has been inserted into the hole. It retrieves the z-coordinates
        of the bottom points of the peg and calculates their distances from the upper surface of the hole.

        Returns:
        - peg_relative_z_mm (np.ndarray): An array of relative z-coordinates in millimeters.
        """
        peg_pts_m = self.peg_abd.get_positions().cpu().numpy().copy()
        peg_bottom_z_m = peg_pts_m[self.peg_bottom_pts_id][:, 2]
        return (peg_bottom_z_m - self.hole_upper_z_m) * 1000

    def _get_peg_pose(self, output_format="quat"):
        R_T_matrix = self.peg_abd.get_transformation_matrix()
        R = R_T_matrix[:3, :3].cpu().detach().numpy()
        t = R_T_matrix[:3, 3].cpu().detach().numpy()
        if output_format == "quat":
            q_R = t3d.quaternions.mat2quat(R)
            return np.around(q_R, decimals=8).tolist(), t
        elif output_format == "euler":
            euler = np.array(t3d.euler.mat2euler(R)) * 180 / np.pi
            return np.around(euler, decimals=5).tolist(), t

    def _sample_points(self, points: np.ndarray, max_points: int):
        if points.shape[0] < max_points:
            selected_indices = np.random.choice(
                points.shape[0], max_points, replace=True
            )
        else:
            selected_indices = np.random.choice(
                points.shape[0], max_points, replace=True
            )
        return points[selected_indices, :]

    def _gen_camera_pose(self, y=0.0, z=0.0, add_random_offset=False):
        camera_theta = (90 + 53.75) / 2 * np.pi / 180  # align to real

        if add_random_offset:
            camera_theta += (np.random.rand() * 2 - 1) * 0.04
            random_offset = (np.random.rand(3) * 2 - 1) * 0.02
        else:
            random_offset = np.zeros(3)
        ####align sim2real#######
        return sapien.Pose(
            p=[
                0.371 + random_offset[0],
                y + 0.02 + random_offset[1],
                z + 0.25 + random_offset[2],
            ],
            q=[np.cos(camera_theta), 0, np.sin(camera_theta), 0],
        )

    def close(self):
        self.ipc_system = None
        return super().close()


class PegInsertionSimMarkerFLowEnvV2(PegInsertionSimEnvV2):
    """
    An environment class that extends PegInsertionSimEnv to incorporate marker flow observations for tactile sensors.

    This class adds functionality for handling marker flow observations from tactile sensors,
    which can be used to simulate realistic sensor noise and dynamics in a peg insertion task.
    """

    def __init__(
        self,
        render_rgb: bool = False,
        marker_interval_range: Tuple[float, float] = (2.0, 2.0),
        marker_rotation_range: float = 0.0,
        marker_translation_range: Tuple[float, float] = (0.0, 0.0),
        marker_pos_shift_range: Tuple[float, float] = (0.0, 0.0),
        marker_random_noise: float = 0.0,
        marker_lose_tracking_probability: float = 0.0,
        normalize: bool = False,
        **kwargs,
    ):
        """
        Initializes a new instance of PegInsertionSimMarkerFLowEnv.

        Args:
        - marker_interval_range (Tuple[float, float]): The range of intervals between marker points in millimeters.
        - marker_rotation_range (float): The range of overall marker rotation in radians.
        - marker_translation_range (Tuple[float, float]): The range of overall marker translation in millimeters.
        - marker_pos_shift_range (Tuple[float, float]): The range of independent marker position shift in millimeters.
        - marker_random_noise (float): The standard deviation of Gaussian noise applied to marker points in pixels.
        - marker_lose_tracking_probability (float): The probability of losing tracking for each marker.
        - normalize (bool): A flag indicating whether to normalize the marker flow observations.
        - **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.render_rgb = render_rgb
        self.sensor_meta_file = kwargs.get("params").tac_sensor_meta_file
        self.marker_interval_range = marker_interval_range
        self.marker_rotation_range = marker_rotation_range
        self.marker_translation_range = marker_translation_range
        self.marker_pos_shift_range = marker_pos_shift_range
        self.marker_random_noise = marker_random_noise
        self.marker_lose_tracking_probability = marker_lose_tracking_probability
        self.normalize = normalize
        self.marker_flow_size = 128

        super(PegInsertionSimMarkerFLowEnvV2, self).__init__(**kwargs)

        self.default_observation.pop("surface_pts")
        self.default_observation["marker_flow"] = np.zeros(
            (2, 2, self.marker_flow_size, 2), dtype=np.float32
        )
        if self.render_rgb:
            self.default_observation["rgb_images"] = np.stack(
                [
                    np.zeros((240, 320, 3), dtype=np.uint8),
                    np.zeros((240, 320, 3), dtype=np.uint8),
                ],
                axis=0,
            )

        self.observation_space = convert_observation_to_space(self.default_observation)

    def _get_sensor_surface_vertices(self) -> list[np.ndarray, np.ndarray]:
        """
        Retrieves the surface vertices of the tactile sensors in the camera coordinate system.

        This method is used to get the current surface vertices of the tactile sensors, which can be used to calculate the relative motion or deformation of the surface in the context of the camera view.

        Returns:
        - vertices (list of np.ndarray): A list containing the surface vertices of the left and right tactile sensors in the camera coordinate system.
        """
        return [
            self.tactile_sensor_1.get_surface_vertices_camera(),
            self.tactile_sensor_2.get_surface_vertices_camera(),
        ]

    def _add_tactile_sensors(self, init_pos_l, init_rot_l, init_pos_r, init_rot_r):
        """
        Initializes and adds two tactile sensors with marker flow capabilities to the simulation environment.

        This method creates two instances of VisionTactileSensorSapienIPC, one for each side (left and right) of the peg insertion mechanism.
        Each sensor is configured with its own initial position, rotation, and marker flow parameters.

        Parameters:
        - init_pos_l (list or np.array): The initial position of the left tactile sensor in meters.
        - init_rot_l (list or np.array): The initial rotation (as a quaternion) of the left tactile sensor.
        - init_pos_r (list or np.array): The initial position of the right tactile sensor in meters.
        - init_rot_r (list or np.array): The initial rotation (as a quaternion) of the right tactile sensor.

        Returns:
        - None
        """
        self.tactile_sensor_1 = VisionTactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_l,
            init_rot=init_rot_l,
            elastic_modulus=self.params.tac_elastic_modulus_l,
            poisson_ratio=self.params.tac_poisson_ratio_l,
            density=self.params.tac_density_l,
            friction=self.params.tac_friction,
            name="tactile_sensor_1",
            marker_interval_range=self.marker_interval_range,
            marker_rotation_range=self.marker_rotation_range,
            marker_translation_range=self.marker_translation_range,
            marker_pos_shift_range=self.marker_pos_shift_range,
            marker_random_noise=self.marker_random_noise,
            marker_lose_tracking_probability=self.marker_lose_tracking_probability,
            normalize=self.normalize,
            marker_flow_size=self.marker_flow_size,
            no_render=self.no_render,
            logger=self.unique_logger,
        )

        self.tactile_sensor_2 = VisionTactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_r,
            init_rot=init_rot_r,
            elastic_modulus=self.params.tac_elastic_modulus_r,
            poisson_ratio=self.params.tac_poisson_ratio_r,
            density=self.params.tac_density_r,
            friction=self.params.tac_friction,
            name="tactile_sensor_2",
            marker_interval_range=self.marker_interval_range,
            marker_rotation_range=self.marker_rotation_range,
            marker_translation_range=self.marker_translation_range,
            marker_pos_shift_range=self.marker_pos_shift_range,
            marker_random_noise=self.marker_random_noise,
            marker_lose_tracking_probability=self.marker_lose_tracking_probability,
            normalize=self.normalize,
            marker_flow_size=self.marker_flow_size,
            no_render=self.no_render,
            logger=self.unique_logger,
        )

    def get_obs(self, info) -> dict:
        """
        Constructs the observation dictionary including marker flow data from the environment's state.

        This method generates the observation that an agent would receive after taking an action in the environment.
        In addition to the standard observations from the parent class, this method includes marker flow data from the tactile sensors.

        Parameters:
        - info (dict): Optional dictionary containing additional environment state information.

        Returns:
        - obs (dict): A dictionary containing the observation data, including marker flow from tactile sensors.
        """
        obs = super().get_obs(info)
        obs.pop("surface_pts")
        obs["marker_flow"] = np.stack(
            [
                self.tactile_sensor_1.gen_marker_flow(),
                self.tactile_sensor_2.gen_marker_flow(),
            ],
            axis=0,
        ).astype(np.float32)
        if self.render_rgb:
            obs["rgb_images"] = np.stack(
                [
                    self.tactile_sensor_1.gen_rgb_image(),
                    self.tactile_sensor_2.gen_rgb_image(),
                ],
                axis=0,
            )

        return obs


class Depth_sensor(sapien_sensor.StereoDepthSensor):
    def __init__(
        self,
        config: sapien_sensor.StereoDepthSensorConfig,
        mount_entity: sapien.Entity,
    ):
        super().__init__(config, mount_entity)
        self.rgb = False
        self.depth = False
        self.point_cloud = False

    def reset(self):
        self.rgb = False
        self.depth = False
        self.point_cloud = False

    def set_pose(self, pose: sapien.Pose) -> None:
        self._mount.set_pose(pose)

    def get_entity(self) -> sapien.Entity:
        return self._mount

    def set_name(self, name: str) -> None:
        self._mount.set_name(name)

    def get_rgb_data(self) -> np.ndarray:
        self.rgb = True
        scene: sapien.Scene = self._mount.get_scene()
        if scene is None:
            raise RuntimeError("Cannot take picture: sensor is not added to scene")
        scene.update_render()
        self._cam_rgb.take_picture()
        return self._cam_rgb.get_picture("Color")[:, :, :3]

    def get_depth_data(self):
        self.depth = True
        self.take_picture(True)
        with suppress_stdout_stderr():
            self.compute_depth()
        # if you want to get lr pic
        # ir_l, ir_r = self.get_ir()
        depth = self.get_depth()
        return depth

    def get_point_cloud(self, gt_point_cloud=False):
        if gt_point_cloud:
            if not self.rgb:
                scene: sapien.Scene = self._mount.get_scene()
                scene.update_render()
                self._cam_rgb.take_picture()
            points_frame = self._cam_rgb.get_picture("Position")[:, :, :3]
            points_frame[:, :, 1] = -points_frame[:, :, 1]
            points_frame[:, :, 2] = -points_frame[:, :, 2]
        else:
            if not self.depth:
                self.take_picture(True)
                self.compute_depth()
            points = self.get_pointcloud()
            points_frame = points.reshape(
                [
                    self.get_config().rgb_resolution[1],
                    self.get_config().rgb_resolution[0],
                    3,
                ]
            )

        return points_frame


if __name__ == "__main__":
    timestep = 0.1
    use_gui = True
    use_render_rgb = True

    log_time = get_time()
    log_folder = Path(os.path.join(track_path, f"Memo/{log_time}"))
    log_dir = Path(os.path.join(log_folder, "main.log"))
    log.remove()
    log.add(log_dir, filter=lambda record: record["extra"]["name"] == "main")
    log.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
        level="INFO",
        filter=lambda record: record["extra"]["name"] == "main",
    )
    test_log = log.bind(name="main")

    params = PegInsertionParams(
        sim_time_step=timestep,
        sim_d_hat=0.1e-3,
        sim_kappa=1e2,
        sim_kappa_affine=1e5,
        sim_kappa_con=1e10,
        sim_eps_d=0,
        sim_eps_v=1e-3,
        sim_solver_newton_max_iters=10,
        sim_solver_cg_max_iters=50,
        sim_solver_cg_error_tolerance=0,
        sim_solver_cg_error_frequency=10,
        ccd_slackness=0.7,
        ccd_thickness=1e-6,
        ccd_tet_inversion_thres=0.0,
        ee_classify_thres=1e-3,
        ee_mollifier_thres=1e-3,
        allow_self_collision=False,
        line_search_max_iters=10,
        ccd_max_iters=100,
        tac_sensor_meta_file="gelsight_mini_e430/meta_file",
        tac_elastic_modulus_l=3.0e5,
        tac_poisson_ratio_l=0.3,
        tac_density_l=1e3,
        tac_elastic_modulus_r=3.0e5,
        tac_poisson_ratio_r=0.3,
        tac_density_r=1e3,
        tac_friction=100,
        # task specific parameters
        gripper_x_offset_mm=0,
        gripper_z_offset_mm=-8,
        indentation_depth_mm=1,
        peg_friction=10,
        hole_friction=1,
    )
    print(params)

    vision_params = {
        "render_mode": "rast",  # "rast" or "rt"
        "vision_type": [
            "rgb",
            "point_cloud",
            "depth",
        ],  # ["rgb", "depth", "point_cloud"], take one, two, or all three.
        # if use point_cloud, the following parameters are needed
        "gt_point_cloud": False,  #  If True is specified, use the ground truth point cloud; otherwise, use the point cloud under render_mode.
        "max_points": 128,  # sample the points from the point cloud
        # if use ray_trace, the following parameters are needed
        # "ray_tracing_denoiser": "optix",
        # "ray_tracing_samples_per_pixel": 4,
    }

    env = PegInsertionSimMarkerFLowEnvV2(
        params=params,
        params_upper_bound=params,
        gui=use_gui,
        step_penalty=1,
        final_reward=10,
        peg_x_max_offset_mm=5,
        peg_y_max_offset_mm=5,
        peg_theta_max_offset_deg=10,
        max_action_mm_deg=np.array([1.0, 1.0, 1.0, 1.0]),
        max_steps=50,
        insertion_depth_mm=1,
        render_rgb=use_render_rgb,
        marker_interval_range=(2.0625, 2.0625),
        marker_rotation_range=0.0,
        marker_translation_range=(0.0, 0.0),
        marker_pos_shift_range=(0.0, 0.0),
        marker_random_noise=0.1,
        marker_lose_tracking_probability=0.0,
        normalize=False,
        peg_dist_z_mm=6.0,
        peg_dist_z_diff_mm=3.0,
        vision_params=vision_params,
        peg_hole_path_file="configs/peg_insertion/to_real_multihole_1shape.txt",
        log_path=log_folder,
        logger=log,
        device="cuda:0",
        no_render=False,
        env_type="test",
    )

    np.set_printoptions(precision=4)

    def visualize_marker_point_flow(o, i, name, save_dir="marker_flow_images"):
        # Create a directory to save images if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        lr_marker_flow = o["marker_flow"]
        l_marker_flow, r_marker_flow = lr_marker_flow[0], lr_marker_flow[1]
        plt.figure(1, (20, 9))
        ax = plt.subplot(1, 2, 1)
        ax.scatter(l_marker_flow[0, :, 0], l_marker_flow[0, :, 1], c="blue")
        ax.scatter(l_marker_flow[1, :, 0], l_marker_flow[1, :, 1], c="red")
        plt.xlim(15, 315)
        plt.ylim(15, 235)
        ax.invert_yaxis()
        ax = plt.subplot(1, 2, 2)
        ax.scatter(r_marker_flow[0, :, 0], r_marker_flow[0, :, 1], c="blue")
        ax.scatter(r_marker_flow[1, :, 0], r_marker_flow[1, :, 1], c="red")
        plt.xlim(15, 315)
        plt.ylim(15, 235)
        ax.invert_yaxis()

        # Save the figure with a filename based on the loop parameter i
        filename = os.path.join(
            save_dir, f"sp-from-sapienipc-{name}-marker_flow_{i}.png"
        )
        plt.savefig(filename)
        plt.close()

    start_time = time.time()
    test_times = 5
    for ii in range(test_times):

        offset_list = []
        offset_list.append([5, -5, 10, 9, 1])
        # offset_list.append([5, 5, -10, 9, 0])
        action_list = []
        action_list.append(
            [[-0.5, 1, -1, -0.2]] * 10
            + [[0.05, 1, 0, 0]] * 18
            + [[0.0, 0, 0, -1]] * 7
            + [[0.0, 0, 0, -1]] * 2
        )
        # action_list.append(
        #     [[-0.5, -1, 1, -0.2]] * 10
        #     + [[0.05, -1, 0, 0]] * 18
        #     + [[0.0, 0, 0, -1]] * 7
        #     + [[0.0, 0, 0, -1]] * 2
        # )

        for offset, actions in zip(offset_list, action_list):
            o, info = env.reset([0,0,0,0,0])
            for k, v in o.items():
                test_log.info(f"{k} : {v.shape}")
            test_log.info(f"timestep: {timestep}")
            test_log.info(f"info : {info}\n")
            if use_gui:
                input("Press Enter to continue...")

            for i, action in enumerate(actions):
                test_log.info(f"action: {action}")
                o, r, d, _, info = env.step(action)
                test_log.info(f"info : ")
                for k, v in info.items():
                    test_log.info(f"{k} : {v}")
                test_log.info(
                    f"gt_offset(mm,deg) : {o['gt_offset']}\n \
                            relative_motion(mm,deg) : {o['relative_motion']}\n"
                )
                test_log.info(f"reward : {r}")
                # visualize_marker_point_flow(o, i, str(offset), save_dir="saved_images")
            if env.viewer is not None:
                while True:
                    if env.viewer.window.key_down("c"):
                        break
                    env.scene.update_render()
                    ipc_update_render_all(env.scene)
                    env.viewer.render()

    end_time = time.time()
    test_log.info(f"{test_times} times cost time: {end_time - start_time}")
