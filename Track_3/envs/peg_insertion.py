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

wp.init()
wp_device = wp.get_preferred_device()


def evaluate_error(info):
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


class PegInsertionSimEnv(gym.Env):
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
        # peg movement
        max_action_mm_deg: np.ndarray = [2.0, 2.0, 4.0],
        max_steps: int = 15,
        z_step_size_mm: float = 0.075,
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
        **kwargs,
    ):
        """
        Initializes a new instance of PegInsertionSimEnv.

        Parameters:
        - step_penalty (float): The penalty applied for each step taken. Defaults to 1.0.
        - final_reward (float): The reward given when the peg is successfully inserted. Defaults to 10.0.
        - peg_hole_path_file (str): The file path to the peg and hole data. Defaults to an empty string.
        - peg_x_max_offset_mm (float): The maximum x-axis offset for the peg in millimeters. Defaults to 5.0.
        - peg_y_max_offset_mm (float): The maximum y-axis offset for the peg in millimeters. Defaults to 5.0.
        - peg_theta_max_offset_deg (float): The maximum rotation offset for the peg in degrees. Defaults to 10.0.
        - max_action_mm_deg (np.ndarray): The maximum action values in millimeters and degrees. Defaults to [2.0, 2.0, 4.0].
        - max_steps (int): The maximum number of steps allowed in an episode. Defaults to 15.
        - z_step_size_mm (float): The size of each step in the z-axis in millimeters. Defaults to 0.075.
        - params (PegInsertionParams): The parameters specific to the peg insertion environment. Defaults to None.
        - params_upper_bound (PegInsertionParams): The upper bound for the parameters. Defaults to None.
        - device (str): The device to use for computations (e.g., "cuda:0"). Defaults to "cuda:0".
        - no_render (bool): A flag indicating whether to render the environment. Defaults to False.
        - log_path (str): The folder path or log file path for logging. Defaults to None.
        - env_type (str): The type of environment (e.g., "train" or "eval"). Defaults to "train".
        - gui (bool): A flag to enable GUI rendering.. Defaults to False.
        - **kwargs: Additional keyword arguments.
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
                f"{self.log_time}_{env_type}_{self.pid}_PegInsertionEnv.log",
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
        super(PegInsertionSimEnv, self).__init__()

        # Initialize environment parameters
        self.step_penalty = step_penalty
        self.final_reward = final_reward
        self.max_steps = max_steps
        self.z_step_size_mm = z_step_size_mm
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
        if self.gui:
            sapien_system = [
                sapien.physx.PhysxCpuSystem(),
                sapien.render.RenderSystem(device=device),
                self.ipc_system,
            ]
        else:
            sapien_system = [sapien.render.RenderSystem(device=device), self.ipc_system]
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

        max_action_mm_deg = np.array(max_action_mm_deg)
        assert max_action_mm_deg.shape == (3,)
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
        """
        Retrieves the default observation of the tactile sensors.

        This method generates the initial observation of the environment, which includes
        the relative motion, ground truth offset, and surface points of the tactile sensors.
        It serves as a baseline for the observations returned by the environment.

        Returns:
        - obs (dict): A dictionary containing the default observation data, including:
            - "relative_motion": An array of zeros representing the initial relative motion.
            - "gt_offset": An array of zeros representing the initial ground truth offset.
            - "surface_pts": A stack of arrays representing the initial surface points of the sensors.
        """

        meta_file = self.params.tac_sensor_meta_file
        meta_file = Path(track_path) / "assets" / meta_file
        with open(meta_file, "r") as f:
            config = json.load(f)
        meta_dir = Path(meta_file).dirname()
        on_surface_np = np.loadtxt(meta_dir / config["on_surface"]).astype(np.int32)
        initial_surface_pts = np.zeros((np.sum(on_surface_np), 3)).astype(float)

        obs = {
            "relative_motion": np.zeros((4,), dtype=np.float32),
            "gt_offset": np.zeros((3,), dtype=np.float32),
            "surface_pts": np.stack([np.stack([initial_surface_pts] * 2)] * 2),
        }
        return obs

    def reset(
        self, offset_mm_deg=None, seed=None, peg_idx: int = None, options=None
    ) -> Tuple[dict, dict]:
        """
        Resets the environment to its initial state.

        This method resets the environment by reinitializing the peg, hole, and tactile sensors.
        It also randomizes the peg's position and orientation within specified bounds if no offset is provided.

        Parameters:
        - offset_mm_deg (list or np.ndarray): An optional offset for the peg's position and orientation in millimeters and degrees.
                                    Should be a list or array of shape (3,) containing x, y offsets in mm and theta offset in degrees.
        - seed (int): An optional seed for random number generation.
        - peg_idx (int): An optional index to select a specific peg-hole pair.
        - options (dict): Additional options for resetting the environment.

        Returns:
        - obs (dict): The initial observation of the environment.
        - info (dict): Additional information about the environment's state.

        """
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

        offset_mm_deg = self._initialize(offset_mm_deg, peg_idx)
        self.unique_logger.info(f"after initialize offset_mm_deg: {offset_mm_deg}")

        self.init_left_surface_pts_m = self.no_contact_surface_mesh[0]
        self.init_right_surface_pts_m = self.no_contact_surface_mesh[1]
        self.init_offset_of_current_episode_mm_deg = offset_mm_deg
        self.current_offset_of_current_episode_mm_deg = offset_mm_deg
        info = self.get_info()
        self.error_evaluation_list = []
        self.error_evaluation_list.append(evaluate_error(info))
        obs = self.get_obs(info)

        return obs, info

    def _initialize(
        self, offset_mm_deg: Union[np.ndarray, None], peg_idx: Union[int, None] = None
    ) -> np.ndarray:
        """
        Initializes the peg and hole entities in the simulation environment.

        This method is responsible for setting up the peg and hole in the scene, applying random or specified offsets,
        and configuring the tactile sensors. It also handles the placement of the peg and hole based on the provided indices or randomly.

        Parameters:
        - offset_mm_deg (Union[np.ndarray, None]): An optional numpy array or None, containing the x and y offsets in millimeters and theta offset in degrees for the peg's position.
                                               Should be of shape (3,) with x and y offsets in mm and theta offset in degrees.
        - peg_idx (Union[int, None]): An optional integer index or None to select a specific peg-hole pair from the list.

        Returns:
        - offset_mm_deg (np.ndarray): The actual offset applied to the peg, including x, y offsets in millimeters and theta offset in degrees.

        """
        # remove all entities except camera
        for e in self.scene.entities:
            if "camera" not in e.name:
                e.remove_from_scene()
        self.ipc_system.rebuild()

        # If in the process of evaluation, select sequentially; if in the process of training, select randomly.
        if peg_idx is None:
            peg_path, hole_path = self.peg_hole_path_list[
                np.random.randint(len(self.peg_hole_path_list))
            ]
        else:
            assert peg_idx < len(self.peg_hole_path_list)
            peg_path, hole_path = self.peg_hole_path_list[peg_idx]

        # get peg and hole path
        asset_dir = Path(track_path) / "assets"
        peg_path = asset_dir / peg_path
        hole_path = asset_dir / hole_path
        self.unique_logger.info("Peg name: " + str(peg_path))

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
            peg = fcl.BVHModel()
            if self.peg_ext == ".msh":
                peg.beginModel(
                    peg_abd.tet_mesh.vertices.shape[0],
                    peg_abd.tet_mesh.surface_triangles.shape[0],
                )
                peg.addSubModel(
                    peg_abd.tet_mesh.vertices, peg_abd.tet_mesh.surface_triangles
                )
            else:
                peg.beginModel(
                    peg_abd.tri_mesh.vertices.shape[0],
                    peg_abd.tri_mesh.surface_triangles.shape[0],
                )
                peg.addSubModel(
                    peg_abd.tri_mesh.vertices, peg_abd.tri_mesh.surface_triangles
                )
            peg.endModel()

            hole = fcl.BVHModel()
            if self.hole_ext == ".msh":
                hole.beginModel(
                    hole_abd.tet_mesh.vertices.shape[0],
                    hole_abd.tet_mesh.surface_triangles.shape[0],
                )
                hole.addSubModel(
                    hole_abd.tet_mesh.vertices, hole_abd.tet_mesh.surface_triangles
                )
            else:
                hole.beginModel(
                    hole_abd.tri_mesh.vertices.shape[0],
                    hole_abd.tri_mesh.surface_triangles.shape[0],
                )
                hole.addSubModel(
                    hole_abd.tri_mesh.vertices, hole_abd.tri_mesh.surface_triangles
                )
            hole.endModel()

            t1 = fcl.Transform()
            peg_fcl = fcl.CollisionObject(peg, t1)
            t2 = fcl.Transform()
            hole_fcl = fcl.CollisionObject(hole, t2)

            while True:
                x_offset_m = (
                    (np.random.rand() * 2 - 1) * self.peg_x_max_offset_mm / 1000
                )
                y_offset_m = (
                    (np.random.rand() * 2 - 1) * self.peg_y_max_offset_mm / 1000
                )
                theta_offset_rad = (
                    (np.random.rand() * 2 - 1)
                    * self.peg_theta_max_offset_deg
                    * np.pi
                    / 180
                )

                R = t3d.euler.euler2mat(0.0, 0.0, theta_offset_rad, axes="rxyz")
                T = np.array([x_offset_m, y_offset_m, 0.0])
                t3 = fcl.Transform(R, T)
                peg_fcl.setTransform(t3)

                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()

                ret = fcl.collide(peg_fcl, hole_fcl, request, result)

                if ret > 0:
                    offset_mm_deg = np.array(
                        [
                            x_offset_m * 1000,
                            y_offset_m * 1000,
                            theta_offset_rad * 180 / np.pi,
                        ]
                    )
                    break
        else:
            x_offset_m, y_offset_m, theta_offset_rad = (
                offset_mm_deg[0] / 1000,
                offset_mm_deg[1] / 1000,
                offset_mm_deg[2] * np.pi / 180,
            )

        # add peg to the scene
        init_pos_m = (
            x_offset_m,
            y_offset_m,
            hole_height_m + 0.1e-3,
        )
        init_theta_offset_rad = theta_offset_rad
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
                math.cos(init_theta_offset_rad) * gripper_x_offset_m + init_pos_m[0],
                math.sin(init_theta_offset_rad) * gripper_x_offset_m + init_pos_m[1],
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
            self.tactile_sensor_1.step()
            self.tactile_sensor_2.step()
            if self.gui:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

        if isinstance(
            self.tactile_sensor_1, VisionTactileSensorSapienIPC
        ) and isinstance(self.tactile_sensor_2, VisionTactileSensorSapienIPC):
            self.tactile_sensor_1.set_reference_surface_vertices_camera()
            self.tactile_sensor_2.set_reference_surface_vertices_camera()
        self.no_contact_surface_mesh = copy.deepcopy(
            self._get_sensor_surface_vertices()
        )

        # Move the peg to create contact between the peg and the hole
        z_distance_m = 0.1e-3 + self.z_step_size_mm / 1000
        pre_insertion_step = max(
            round((z_distance_m / 1e-3) / self.params.sim_time_step), 1
        )
        pre_insertion_speed = (
            z_distance_m / pre_insertion_step / self.params.sim_time_step
        )
        for _ in range(pre_insertion_step):
            self.tactile_sensor_1.set_active_v([0, 0, -pre_insertion_speed])
            self.tactile_sensor_2.set_active_v([0, 0, -pre_insertion_speed])

            with suppress_stdout_stderr():
                self.hole_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
                )  # hole stays static
                self.ipc_system.step()
            self.tactile_sensor_1.step()
            self.tactile_sensor_2.step()
            if self.gui:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

        # mem log
        monitor_process_memory_once(self.pid, self.unique_logger)
        monitor_process_gpu_memory(self.pid, self.unique_logger)

        return offset_mm_deg

    def _add_tactile_sensors(self, init_pos_l, init_rot_l, init_pos_r, init_rot_r):
        """
        Initializes and adds two tactile sensors to the Sapien scene.

        This method creates two instances of TactileSensorSapienIPC, one for each side (left and right) of the peg insertion mechanism.
        Each sensor is configured with its own initial position and rotation.

        Parameters:
        - init_pos_l (list or np.array): The initial position of the left tactile sensor in meters.
        - init_rot_l (list or np.array): The initial rotation (as a quaternion) of the left tactile sensor.
        - init_pos_r (list or np.array): The initial position of the right tactile sensor in meters.
        - init_rot_r (list or np.array): The initial rotation (as a quaternion) of the right tactile sensor.

        Returns:
        - None
        """

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
        action_z_mm = -self.z_step_size_mm

        action_theta_deg = action_mm_deg[2]

        self.current_offset_of_current_episode_mm_deg[0] += action_x_mm
        self.current_offset_of_current_episode_mm_deg[1] += action_y_mm
        self.current_offset_of_current_episode_mm_deg[2] += action_theta_deg
        self.sensor_grasp_center_current_mm_deg[0] += action_x_mm
        self.sensor_grasp_center_current_mm_deg[1] += action_y_mm
        self.sensor_grasp_center_current_mm_deg[2] += action_theta_deg
        self.sensor_grasp_center_current_mm_deg[3] += action_z_mm

        action_sim_mm_deg = np.array([action_x_mm, action_y_mm, action_theta_deg])
        sensor_grasp_center_m = (
            self.tactile_sensor_1.current_pos + self.tactile_sensor_2.current_pos
        ) / 2

        x_m = action_sim_mm_deg[0] / 1000
        y_m = action_sim_mm_deg[1] / 1000
        theta_rad = action_sim_mm_deg[2] * np.pi / 180

        action_substeps = max(
            1, round((max(abs(x_m), abs(y_m)) / 5e-3) / self.params.sim_time_step)
        )
        action_substeps = max(
            action_substeps, round((abs(theta_rad) / 0.2) / self.params.sim_time_step)
        )
        v_x = x_m / self.params.sim_time_step / action_substeps
        v_y = y_m / self.params.sim_time_step / action_substeps
        v_theta = theta_rad / self.params.sim_time_step / action_substeps

        for _ in range(action_substeps):
            self.tactile_sensor_1.set_active_v_r(
                [v_x, v_y, 0],
                sensor_grasp_center_m,
                (0, 0, 1),
                v_theta,
            )
            self.tactile_sensor_2.set_active_v_r(
                [v_x, v_y, 0],
                sensor_grasp_center_m,
                (0, 0, 1),
                v_theta,
            )
            with suppress_stdout_stderr():
                self.hole_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
                )  # hole stays static
                self.ipc_system.step()
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

        z_m = -self.z_step_size_mm / 1000
        z_substeps = max(1, round(abs(z_m) / 5e-3 / self.params.sim_time_step))
        v_z = z_m / self.params.sim_time_step / z_substeps
        for _ in range(z_substeps):
            self.tactile_sensor_1.set_active_v(
                [0, 0, v_z],
            )
            self.tactile_sensor_2.set_active_v(
                [0, 0, v_z],
            )
            with suppress_stdout_stderr():
                self.hole_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
                )  # hole stays static
                self.ipc_system.step()
            state1 = self.tactile_sensor_1.step()
            state2 = self.tactile_sensor_2.step()
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
        x_offset_mm, y_offset_mm, theta_offset_deg = (
            self.current_offset_of_current_episode_mm_deg
        )
        if (
            np.abs(x_offset_mm) > 12 + 1e-5
            or np.abs(y_offset_mm) > 12 + 1e-5
            or np.abs(theta_offset_deg) > 15 + 1e-5
        ):
            info["error_too_large"] = True

        peg_relative_z_mm = self._get_peg_relative_z_mm()
        info["peg_relative_z_mm"] = peg_relative_z_mm
        info["is_success"] = False
        if (
            not info["error_too_large"]
            and not info["too_many_steps"]
            and not info["tactile_movement_too_large"]
            and self.current_episode_elapsed_steps * self.z_step_size_mm > 0.35
            and np.sum(peg_relative_z_mm < -0.3) == peg_relative_z_mm.shape[0]
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
        The observation includes the ground truth offset, relative motion, and surface points of the tactile sensors.

        Parameters:
        - info (dict): A dictionary containing the environment's state information, obtained from `get_info`.

        Returns:
        - obs (dict): A dictionary containing the observation data.
        """

        obs_dict = dict()

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

        self.error_evaluation_list.append(evaluate_error(info))

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

    def close(self):
        self.ipc_system = None
        return super().close()


class PegInsertionSimMarkerFLowEnv(PegInsertionSimEnv):
    """
    An environment class that extends PegInsertionSimEnv to incorporate marker flow observations for tactile sensors.

    This class adds functionality for handling marker flow observations from tactile sensors,
    which can be used to simulate realistic sensor noise and dynamics in a peg insertion task.
    """

    def __init__(
        self,
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
        self.sensor_meta_file = kwargs.get("params").tac_sensor_meta_file
        self.marker_interval_range = marker_interval_range
        self.marker_rotation_range = marker_rotation_range
        self.marker_translation_range = marker_translation_range
        self.marker_pos_shift_range = marker_pos_shift_range
        self.marker_random_noise = marker_random_noise
        self.marker_lose_tracking_probability = marker_lose_tracking_probability
        self.normalize = normalize
        self.marker_flow_size = 128

        super(PegInsertionSimMarkerFLowEnv, self).__init__(**kwargs)

        self.default_observation.pop("surface_pts")
        self.default_observation["marker_flow"] = np.zeros(
            (2, 2, self.marker_flow_size, 2), dtype=np.float32
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
        return obs


if __name__ == "__main__":
    timestep = 0.1
    use_gui = False

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
        tac_sensor_meta_file="tac_sensor_meta/gelsight_mini_e430/meta_file",
        tac_elastic_modulus_l=3.0e5,
        tac_poisson_ratio_l=0.3,
        tac_density_l=1e3,
        tac_elastic_modulus_r=3.0e5,
        tac_poisson_ratio_r=0.3,
        tac_density_r=1e3,
        tac_friction=100,
        # task specific parameters
        gripper_x_offset_mm=0,
        gripper_z_offset_mm=-4,
        indentation_depth_mm=1,
        peg_friction=10,
        hole_friction=1,
    )
    print(params)

    env = PegInsertionSimMarkerFLowEnv(
        params=params,
        params_upper_bound=params,
        gui=use_gui,
        step_penalty=1,
        final_reward=10,
        peg_x_max_offset_mm=5.0,
        peg_y_max_offset_mm=5.0,
        peg_theta_max_offset_deg=10.0,
        max_action_mm_deg=np.array([1.0, 1.0, 1.0]),
        max_steps=10,
        z_step_size_mm=0.5,
        marker_interval_range=(2.0625, 2.0625),
        marker_rotation_range=0.0,
        marker_translation_range=(0.0, 0.0),
        marker_pos_shift_range=(0.0, 0.0),
        marker_random_noise=0.1,
        marker_lose_tracking_probability=0.0,
        normalize=False,
        peg_hole_path_file="configs/peg_insertion/3shape_1.5mm.txt",
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
        offset_list.append([4, 0, 4])
        # offset_list.append([0, 4, -4])
        action_list = []
        action_list.append([[-0.5, 0, -0.25]] * 4 + [[0.0, 0.0, 0.0]] * 4)
        # action_list.append([[0.0, -0.5, 0.25]] * 4 + [[0.0, 0.0, 0.0]] * 4)
        for offset, actions in zip(offset_list, action_list):
            o, info = env.reset(offset, peg_idx=1)
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
