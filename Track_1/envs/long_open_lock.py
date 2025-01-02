import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

import copy
import time
from typing import Tuple


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
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
from utils.gym_env_utils import convert_observation_to_space
from utils.sapienipc_utils import build_sapien_entity_ABD

from loguru import logger as log
from utils.mem_monitor import *

wp.init()
wp_device = wp.get_preferred_device()


class LongOpenLockParams(CommonParams):
    def __init__(
        self,
        key_lock_path_file: str = "",
        key_friction: float = 1.0,
        lock_friction: float = 1.0,
        indentation_depth_mm: float = 0.5,
        **kwargs,
    ):
        """
        A class to store parameters for the LongOpenLock environment.

        Parameters:
        - key_lock_path_file (str): The file path to the key and lock path data.
        - key_friction (float): The friction coefficient for the key.
        - lock_friction (float): The friction coefficient for the lock.
        - indentation_depth_mm (float): The depth of the gripper indentation in millimeters.
        - **kwargs: Additional keyword arguments inherited from CommonParams.
        """
        super().__init__(**kwargs)
        self.key_lock_path_file = key_lock_path_file
        self.indentation_depth_mm = indentation_depth_mm
        self.key_friction = key_friction
        self.lock_friction = lock_friction


class LongOpenLockSimEnv(gym.Env):
    def __init__(
        self,
        # reward
        step_penalty: float = 1.0,
        final_reward: float = 10.0,
        # key position
        key_x_max_offset_mm: float = 10.0,
        key_y_max_offset_mm: float = 0.0,
        key_z_max_offset_mm: float = 0.0,
        sensor_offset_x_range_len_mm: float = 0.0,
        sensor_offset_z_range_len_mm: float = 0.0,
        # key movement
        max_action_mm: np.ndarray = [4.0, 2.0],
        max_steps: int = 100,
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
        A simulation environment for a long open lock mechanism.

        Parameters:
        - step_penalty (float): The penalty for each step taken.
        - final_reward (float): The reward for successfully opening the lock.
        - key_x_max_offset_mm (float): The maximum x-axis offset for the key in millimeters.
        - key_y_max_offset_mm (float): The maximum y-axis offset for the key in millimeters.
        - key_z_max_offset_mm (float): The maximum z-axis offset for the key in millimeters.
        - sensor_offset_x_range_len_mm (float): The range length of x-axis offset for the sensor in millimeters.
        - sensor_offset_z_range_len_mm (float): The range length of z-axis offset for the sensor in millimeters.
        - max_action_mm (np.ndarray): The maximum action values in millimeters.
        - max_steps (int): The maximum number of steps allowed.
        - params (LongOpenLockParams): The parameters object.
        - params_upper_bound (LongOpenLockParams): The upper bound for the parameters.
        - device (str): The device to use for computations (e.g., "cuda:0").
        - no_render (bool): A flag to disable rendering.
        - log_folder (str): The folder path for logging.
        - env_type (str): The type of environment (e.g., "train").
        - gui (bool): A flag to enable GUI rendering.
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
                f"{self.log_time}_{env_type}_{self.pid}_OpenLockEnv.log",
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
        super(LongOpenLockSimEnv, self).__init__()

        self.no_render = no_render
        self.index = None
        self.step_penalty = step_penalty
        self.final_reward = final_reward
        self.max_steps = max_steps
        self.max_action_mm = np.array(max_action_mm)
        assert self.max_action_mm.shape == (2,)

        self.key_x_max_offset_mm = key_x_max_offset_mm
        self.key_y_max_offset_mm = key_y_max_offset_mm
        self.key_z_max_offset_mm = key_z_max_offset_mm
        self.sensor_offset_x_range_len_mm = sensor_offset_x_range_len_mm
        self.sensor_offset_z_range_len_mm = sensor_offset_z_range_len_mm

        self.current_episode_elapsed_steps = 0
        self.current_episode_over = False
        self.sensor_grasp_center_init_m = np.array([0, 0, 0])
        self.sensor_grasp_center_current_m = self.sensor_grasp_center_init_m

        if params is None:
            self.params_lb = LongOpenLockParams()
        else:
            self.params_lb = copy.deepcopy(params)
        if params_upper_bound is None:
            self.params_ub = copy.deepcopy(self.params_lb)
        else:
            self.params_ub = copy.deepcopy(params_upper_bound)
        self.params: LongOpenLockParams = randomize_params(
            self.params_lb, self.params_ub
        )

        key_lock_path_file = Path(track_path) / self.params.key_lock_path_file
        self.key_lock_path_list = []
        with open(key_lock_path_file, "r") as f:
            for l in f.readlines():
                self.key_lock_path_list.append(
                    [ss.strip() for ss in l.strip().split(",")]
                )

        self.init_left_surface_pts_m = None
        self.init_right_surface_pts_m = None

        ######## Create system ########
        ipc_system_config = IPCSystemConfig()
        # memory config
        ipc_system_config.max_scenes = 1
        ipc_system_config.max_surface_primitives_per_scene = 1 << 11
        ipc_system_config.max_blocks = 1000000
        # scene config
        ipc_system_config.time_step = self.params.sim_time_step
        ipc_system_config.gravity = wp.vec3(0, 0, 0)
        ipc_system_config.d_hat = self.params.sim_d_hat  # 2e-4
        ipc_system_config.eps_d = self.params.sim_eps_d  # 1e-3
        ipc_system_config.eps_v = self.params.sim_eps_v  # 1e-3
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
        # ipc_system_config.allow_self_collision = False

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

        # set device
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

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.default_observation, _ = self.reset()
        self.observation_space = convert_observation_to_space(self.default_observation)

    def evaluate_error(self, info, error_scale=500) -> float:
        """
        Evaluates the error based on the current state of the environment.

        This function calculates an error value that represents the difference between the key and the lock.
        The error is calculated based on the positions of the key and lock points, and it is scaled by the `error_scale` parameter.

        Parameters:
        - info (dict): A dictionary containing the current state information, including the positions of the key and lock points.
        - error_scale (int): A scaling factor for the error calculation. Defaults to 500.

        Returns:
        - error_sum (float): The calculated error value.
        """
        error_sum = 0
        key1_pts_center_m = info["key1_pts_m"].mean(0)  # m
        key2_pts_center_m = info["key2_pts_m"].mean(0)
        key1_pts_max_m = info["key1_pts_m"].max(0)
        key2_pts_max_m = info["key2_pts_m"].max(0)
        lock1_pts_center_m = info["lock1_pts_m"].mean(0)
        lock2_pts_center_m = info["lock2_pts_m"].mean(0)

        error_sum += abs(key1_pts_center_m[0] - lock1_pts_center_m[0])  # x direction
        error_sum += abs(key2_pts_center_m[0] - lock2_pts_center_m[0])
        # print(f"reward start: {reward}")
        # z_offset
        if self.index == 0 or self.index == 2:
            if key1_pts_max_m[0] < 0.046 and key2_pts_max_m[0] < 0.046:
                # if key is inside the lock, then encourage it to fit in to the holes
                error_sum += abs(
                    0.037 - key1_pts_center_m[2]
                )  # must be constrained in both directions
                error_sum += abs(
                    0.037 - key2_pts_center_m[2]
                )  # otherwise the policy would keep lifting the key
                # and smooth the error to avoid sudden change
            else:
                # else, align it with the hole
                error_sum += abs(key1_pts_center_m[2] - 0.030)
                error_sum += abs(key2_pts_center_m[2] - 0.030)
                pass
        if self.index == 1:
            if key1_pts_max_m[0] < 0.052 and key2_pts_max_m[0] < 0.052:
                # if key is inside the lock, then encourage it to fit in to the holes
                error_sum += abs(
                    0.037 - key1_pts_center_m[2]
                )  # must be constrained in both directions
                error_sum += abs(
                    0.037 - key2_pts_center_m[2]
                )  # otherwise the policy would keep lifting the key
                # and smooth the error to avoid sudden change
            else:
                # else, align it with the hole
                error_sum += abs(key1_pts_center_m[2] - 0.030)
                error_sum += abs(key2_pts_center_m[2] - 0.030)
                pass
        if self.index == 3:
            if key1_pts_max_m[0] < 0.062 and key2_pts_max_m[0] < 0.062:
                # if key is inside the lock, then encourage it to fit in to the holes
                error_sum += abs(
                    0.037 - key1_pts_center_m[2]
                )  # must be constrained in both directions
                error_sum += abs(
                    0.037 - key2_pts_center_m[2]
                )  # otherwise the policy would keep lifting the key
                # and smooth the error to avoid sudden change
            else:
                # else, align it with the hole
                error_sum += abs(key1_pts_center_m[2] - 0.030)
                error_sum += abs(key2_pts_center_m[2] - 0.030)
                pass

        # y_offset
        error_sum += abs(key1_pts_center_m[1])
        error_sum += abs(key2_pts_center_m[1])
        error_sum *= error_scale
        return error_sum

    def seed(self, seed=None):
        if seed is None:
            seed = (int(time.time() * 1000) % 10000 * os.getpid()) % 2**30
        np.random.seed(seed)

    def reset(
        self, offset_mm=None, seed=None, key_idx: int = None, options=None
    ) -> Tuple[dict, dict]:
        """
        Resets the environment to its initial state.

        This function resets the environment by reinitializing the key, lock, and sensors.
        It also randomizes the key's position within specified bounds.

        Parameters:
        - offset_mm (tuple): An optional offset in millimeters for the key's position.
        - seed (int): An optional seed for random number generation.
        - key_idx (int): An optional index to select a specific key-lock pair.
        - options (dict): Additional options for resetting the environment.

        Returns:
        - obs (dict): The initial observation of the environment.
        - info (dict): Additional information about the environment's state.
        """

        print(self.env_type + " reset once")
        self.unique_logger.info(
            "*************************************************************"
        )
        self.unique_logger.info(self.env_type + " reset once")

        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.params = randomize_params(self.params_lb, self.params_ub)
        self.current_episode_elapsed_steps = 0
        self.current_episode_over = False

        self.initialize(key_offset_mm=offset_mm, key_idx=key_idx)
        self.init_left_surface_pts_m = self.no_contact_surface_mesh[0]
        self.init_right_surface_pts_m = self.no_contact_surface_mesh[1]
        self.error_evaluation_list = []
        info = self.get_info()
        self.error_evaluation_list.append(self.evaluate_error(info))

        return self.get_obs(info), {}

    def initialize(self, key_offset_mm=None, key_idx: int = None):
        """
        Initializes the environment by setting up the key, lock, and sensors.

        This function is responsible for creating the key and lock entities in the simulation scene,
        positioning them according to the provided offsets, and initializing the tactile sensors.

        Parameters:
        - key_offset_mm (tuple): An optional tuple containing the x, y, z offset in millimeters for the key's position.
        - key_idx (int): An optional index to select a specific key-lock pair from the predefined list.
        """

        for e in self.scene.entities:
            if "camera" not in e.name:
                e.remove_from_scene()
        self.ipc_system.rebuild()
        if key_idx is None:
            self.index = np.random.randint(len(self.key_lock_path_list))
            key_path, lock_path = self.key_lock_path_list[self.index]
        else:
            assert key_idx < len(self.key_lock_path_list)
            self.index = key_idx
            key_path, lock_path = self.key_lock_path_list[self.index]

        asset_dir = Path(track_path) / "assets"
        key_path = asset_dir / key_path
        lock_path = asset_dir / lock_path

        if key_offset_mm is None:
            if self.index == 0:
                x_offset = np.random.rand() * self.key_x_max_offset_mm + 46 - 5

            elif self.index == 1:
                x_offset = np.random.rand() * self.key_x_max_offset_mm + 52 - 5
            elif self.index == 2:
                x_offset = np.random.rand() * self.key_x_max_offset_mm + 46 - 5
            elif self.index == 3:
                x_offset = np.random.rand() * self.key_x_max_offset_mm + 62 - 5

            y_offset = (np.random.rand() * 2 - 1) * self.key_y_max_offset_mm
            z_offset = (np.random.rand() * 2 - 1) * self.key_z_max_offset_mm
            key_offset_mm = (x_offset, y_offset, z_offset)
            print(
                "index=",
                self.index,
                "keyoffset_mm=",
                key_offset_mm,
            )
            self.unique_logger.info(f"index={self.index}, keyoffset_mm={key_offset_mm}")
        else:
            x_offset_mm, y_offset_mm, z_offset_mm = tuple(key_offset_mm)
            if self.index == 0:
                x_offset_mm += 46
            elif self.index == 1:
                x_offset_mm += 52
            elif self.index == 2:
                x_offset_mm += 46
            elif self.index == 3:
                x_offset_mm += 62
            key_offset_mm = (x_offset_mm, y_offset_mm, z_offset_mm)
            print(
                "index=",
                self.index,
                "keyoffset_mm=",
                key_offset_mm,
            )
            self.unique_logger.info(f"index={self.index}, keyoffset_mm={key_offset_mm}")

        key_offset_m = [value / 1000 for value in key_offset_mm]

        with suppress_stdout_stderr():
            self.key_entity, key_abd = build_sapien_entity_ABD(
                key_path,
                density=500.0,
                color=[1.0, 0.0, 0.0, 0.95],
                friction=self.params.key_friction,
                no_render=self.no_render,
            )
        self.key_abd = key_abd
        self.key_entity.set_pose(sapien.Pose(p=key_offset_m, q=[0.7071068, 0, 0, 0]))
        self.scene.add_entity(self.key_entity)

        with suppress_stdout_stderr():
            self.lock_entity, lock_abd = build_sapien_entity_ABD(
                lock_path,
                density=500.0,
                color=[0.0, 0.0, 1.0, 1],
                friction=self.params.lock_friction,
                no_render=self.no_render,
            )
        self.hold_abd = lock_abd
        self.scene.add_entity(self.lock_entity)

        sensor_x_mm = np.random.rand() * self.sensor_offset_x_range_len_mm
        sensor_x_mm = sensor_x_mm * np.random.choice([-1, 1])
        sensor_x_m = sensor_x_mm / 1e3  # mm -> m
        sensor_z_mm = np.random.rand() * self.sensor_offset_z_range_len_mm
        sensor_z_mm = sensor_z_mm * np.random.choice([-1, 1])
        sensor_z_m = sensor_z_mm / 1e3  # mm -> m
        if self.index == 0 or self.index == 2:
            init_pos_l_m = np.array(
                [
                    key_offset_m[0] + 0.07 + sensor_x_m,
                    key_offset_m[1] - (6 * 1e-3 / 2 + 0.002 + 0.0005),
                    key_offset_m[2] + 0.016 + sensor_z_m,
                ]
            )
            init_rot_l = np.array([0.7071068, -0.7071068, 0, 0])

            init_pos_r_m = np.array(
                [
                    key_offset_m[0] + 0.07 + sensor_x_m,
                    key_offset_m[1] + 6 * 1e-3 / 2 + 0.002 + 0.0005,
                    key_offset_m[2] + 0.016 + sensor_z_m,
                ]
            )
            init_rot_r = np.array([0.7071068, 0.7071068, 0, 0])

        if self.index == 1:
            init_pos_l_m = np.array(
                [
                    key_offset_m[0] + 0.075 + sensor_x_m,
                    key_offset_m[1] - (6 * 1e-3 / 2 + 0.002 + 0.0005),
                    key_offset_m[2] + 0.016 + sensor_z_m,
                ]
            )
            init_rot_l = np.array([0.7071068, -0.7071068, 0, 0])

            init_pos_r_m = np.array(
                [
                    key_offset_m[0] + 0.075 + sensor_x_m,
                    key_offset_m[1] + 6 * 1e-3 / 2 + 0.002 + 0.0005,
                    key_offset_m[2] + 0.016 + sensor_z_m,
                ]
            )
            init_rot_r = np.array([0.7071068, 0.7071068, 0, 0])

        if self.index == 3:
            init_pos_l_m = np.array(
                [
                    key_offset_m[0] + 0.08 + sensor_x_m,
                    key_offset_m[1] - (6 * 1e-3 / 2 + 0.002 + 0.0005),
                    key_offset_m[2] + 0.016 + sensor_z_m,
                ]
            )
            init_rot_l = np.array([0.7071068, -0.7071068, 0, 0])

            init_pos_r_m = np.array(
                [
                    key_offset_m[0] + 0.08 + sensor_x_m,
                    key_offset_m[1] + 6 * 1e-3 / 2 + 0.002 + 0.0005,
                    key_offset_m[2] + 0.016 + sensor_z_m,
                ]
            )
            init_rot_r = np.array([0.7071068, 0.7071068, 0, 0])

        self.sensor_grasp_center_init_m = np.array(
            [
                key_offset_m[0] + 0.0175 + 0.032 + sensor_x_m,
                key_offset_m[1],
                key_offset_m[2] + 0.016 + sensor_z_m,
            ]
        )
        self.sensor_grasp_center_current_m = self.sensor_grasp_center_init_m.copy()

        with suppress_stdout_stderr():
            self.add_tactile_sensors(init_pos_l_m, init_rot_l, init_pos_r_m, init_rot_r)

        if self.gui:
            self.viewer = viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_pose(
                sapien.Pose(
                    [-0.0877654, 0.0921954, 0.186787],
                    [0.846142, 0.151231, 0.32333, -0.395766],
                )
            )
            pause = True
            while pause:
                if self.viewer.window.key_down("c"):
                    pause = False
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

        grasp_step = max(
            round(
                (0.5 + self.params.indentation_depth_mm)
                / 1000
                / 2e-3
                / self.params.sim_time_step
            ),
            1,
        )
        grasp_speed = (
            (0.5 + self.params.indentation_depth_mm)
            / 1000
            / grasp_step
            / self.params.sim_time_step
        )

        for _ in range(grasp_step):
            self.tactile_sensor_1.set_active_v([0, grasp_speed, 0])
            self.tactile_sensor_2.set_active_v([0, -grasp_speed, 0])
            with suppress_stdout_stderr():
                self.hold_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
                )  # hole stays static
                self.ipc_system.step()

            self.tactile_sensor_1.step()
            self.tactile_sensor_2.step()
            if self.gui:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

        if isinstance(self.tactile_sensor_1, VisionTactileSensorSapienIPC):
            if isinstance(self.tactile_sensor_2, VisionTactileSensorSapienIPC):
                self.tactile_sensor_1.set_reference_surface_vertices_camera()
                self.tactile_sensor_2.set_reference_surface_vertices_camera()
        self.no_contact_surface_mesh = copy.deepcopy(
            self._get_sensor_surface_vertices()
        )

        # mem log
        monitor_process_memory_once(self.pid, self.unique_logger)
        monitor_process_gpu_memory(self.pid, self.unique_logger)

    def add_tactile_sensors(self, init_pos_l, init_rot_l, init_pos_r, init_rot_r):
        """
        Initializes and adds two tactile sensors to the simulation environment.

        This function creates two instances of the TactileSensorSapienIPC class, one for each side (left and right) of the lock mechanism.
        Each sensor is configured with its own initial position and rotation.

        Parameters:
        - init_pos_l (list or np.array): The initial position of the left tactile sensor in meters.
        - init_rot_l (list or np.array): The initial rotation (as a quaternion) of the left tactile sensor.
        - init_pos_r (list or np.array): The initial position of the right tactile sensor in meters.
        - init_rot_r (list or np.array): The initial rotation (as a quaternion) of the right tactile sensor.
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

        This function is called to execute one step of the simulation, where the action is applied to the environment.
        It updates the environment state, calculates the reward, and checks for termination conditions.

        Parameters:
        - action (np.ndarray): A 2D array representing the action to be taken, scaled to millimeters.

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
        action_mm = np.array(action).flatten() * self.max_action_mm
        self.unique_logger.info(f"action_mm: {action_mm}")
        action_m = action_mm / 1000
        self._sim_step(action_m)

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

    def _sim_step(self, action_m):
        """
        Executes a single simulation step with the given action vector.

        This function takes an action vector, converts it into velocities for the tactile sensors,
        and steps the simulation accordingly. It's a helper function used by `step` to perform the actual simulation step.

        Parameters:
        - action_m (np.ndarray): A numpy array representing the action in meters [x, z].

        Returns:
        - None
        """
        substeps = max(
            1, round(np.max(np.abs(action_m)) / 2e-3 / self.params.sim_time_step)
        )
        v = action_m / substeps / self.params.sim_time_step

        # Convert 2D action [x,z] directly to 3D velocity [-x,0,-z]
        v_3d = np.array([-v[0], 0, -v[1]])

        for _ in range(substeps):
            self.tactile_sensor_1.set_active_v(v_3d)
            self.tactile_sensor_2.set_active_v(v_3d)
            with suppress_stdout_stderr():
                self.hold_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
                )
                self.ipc_system.step()
            self.tactile_sensor_1.step()
            self.tactile_sensor_2.step()
            self.sensor_grasp_center_current_m = (
                self.tactile_sensor_1.get_pose()[0]
                + self.tactile_sensor_2.get_pose()[0]
            ) / 2

            if self.gui:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

    def get_info(self) -> dict:
        """
        Retrieves additional information about the environment's state.

        This function collects various metrics and data points that describe the current state of the simulation,
        including the positions of the key and lock points, surface differences, and success criteria.

        Returns:
        - info (dict): A dictionary containing the environment's state information.
        """
        info = {"steps": self.current_episode_elapsed_steps}

        key_pts_m = self.key_abd.get_positions().cpu().numpy().copy()
        lock_pts_m = self.hold_abd.get_positions().cpu().numpy().copy()
        if self.index == 0:
            key1_idx = np.array([16, 17, 18, 19])
            key2_idx = np.array([24, 25, 26, 27])
            key_side_index = np.array([1, 3, 30, 31])
            lock1_idx = np.array([2, 3, 6, 7])
            lock2_idx = np.array([4, 5, 30, 31])
            lock_side_index = np.array([10, 11, 9, 13])
        elif self.index == 1:
            key1_idx = np.array([20, 21, 22, 23])
            key2_idx = np.array([28, 29, 30, 31])
            key_side_index = np.array([0, 2, 4, 5])
            lock1_idx = np.array([0, 1, 6, 7])
            lock2_idx = np.array([30, 31, 2, 3])
            lock_side_index = np.array([8, 9, 11, 13])
        elif self.index == 2:
            key1_idx = np.array([4, 5, 6, 7])
            key2_idx = np.array([12, 13, 14, 15])
            key_side_index = np.array([18, 19, 20, 21])
            lock1_idx = np.array([6, 7, 2, 3])
            lock2_idx = np.array([4, 5, 30, 31])
            lock_side_index = np.array([10, 9, 11, 13])
        elif self.index == 3:
            key1_idx = np.array([8, 9, 10, 11])
            key2_idx = np.array([16, 17, 18, 19])
            key_side_index = np.array([30, 31, 32, 33])
            lock1_idx = np.array([2, 3, 10, 11])
            lock2_idx = np.array([6, 7, 8, 9])
            lock_side_index = np.array([12, 13, 15, 17])

        key1_pts_m = key_pts_m[key1_idx]  # mm
        key2_pts_m = key_pts_m[key2_idx]
        key_side_pts_m = key_pts_m[key_side_index]
        lock1_pts_m = lock_pts_m[lock1_idx]
        lock2_pts_m = lock_pts_m[lock2_idx]
        lock_side_pts_m = lock_pts_m[lock_side_index]

        info["key1_pts_m"] = key1_pts_m
        info["key2_pts_m"] = key2_pts_m
        info["key_side_pts_m"] = key_side_pts_m
        info["lock1_pts_m"] = lock1_pts_m
        info["lock2_pts_m"] = lock2_pts_m
        info["lock_side_pts_m"] = lock_side_pts_m

        observation_left_surface_pts_m, observation_right_surface_pts_m = (  # m
            self._get_sensor_surface_vertices()
        )
        l_diff_m = np.mean(
            np.sqrt(
                np.sum(
                    (self.init_left_surface_pts_m - observation_left_surface_pts_m)
                    ** 2,
                    axis=-1,
                )
            )
        )
        r_diff_m = np.mean(
            np.sqrt(
                np.sum(
                    (self.init_right_surface_pts_m - observation_right_surface_pts_m)
                    ** 2,
                    axis=-1,
                )
            )
        )
        info["surface_diff_m"] = np.array([l_diff_m, r_diff_m])
        info["tactile_movement_too_large"] = False
        if l_diff_m > 1.5e-3 or r_diff_m > 1.5e-3:
            info["tactile_movement_too_large"] = True
        info["relative_motion_mm"] = 1e3 * (  # mm
            self.sensor_grasp_center_current_m - self.sensor_grasp_center_init_m
        )
        info["error_too_large"] = False
        if (
            np.abs(info["key1_pts_m"].mean(0)[1]) > 0.01
            or np.abs(info["key2_pts_m"].mean(0)[1]) > 0.01
            or info["key1_pts_m"].mean(0)[2] > 0.045
            or info["key2_pts_m"].mean(0)[2] > 0.045
            or info["key1_pts_m"].mean(0)[2] < 0.015
            or info["key2_pts_m"].mean(0)[2] < 0.015
            or info["key1_pts_m"].mean(0)[0] > 0.110
            or info["key2_pts_m"].mean(0)[0] > 0.110
        ):
            info["error_too_large"] = True

        info["is_success"] = False
        if (
            key1_pts_m[:, 0].max() < info["lock_side_pts_m"].mean(0)[0]
            and key1_pts_m[:, 0].min() > 0
            and key2_pts_m[:, 0].max() < info["lock_side_pts_m"].mean(0)[0]
            and key2_pts_m[:, 0].min() > 0
            and np.abs(key1_pts_m[:, 1].mean()) < 0.002
            and np.abs(key2_pts_m[:, 1].mean()) < 0.002
            and key1_pts_m[:, 2].min() > 0.033
            and key1_pts_m[:, 2].max() < 0.04
            and key2_pts_m[:, 2].min() > 0.033
            and key2_pts_m[:, 2].max() < 0.04
        ):
            info["is_success"] = True
        return info

    def get_obs(self, info) -> dict:
        """
        Constructs the observation dictionary from the environment's state information.

        This function takes the state information and processes it to create an observation that can be used by an agent or algorithm.
        The observation includes surface points, key and lock points, and other relevant data.

        Parameters:
        - info (dict): A dictionary containing the environment's state information, typically obtained from `get_info`.

        Returns:
        - obs_dict (dict): A dictionary containing the observation data.
        """

        observation_left_surface_pts_m, observation_right_surface_pts_m = (
            self._get_sensor_surface_vertices()
        )
        obs_dict = {
            "surface_pts": np.stack(
                [
                    np.stack(
                        [self.init_left_surface_pts_m, observation_left_surface_pts_m]
                    ),
                    np.stack(
                        [self.init_right_surface_pts_m, observation_right_surface_pts_m]
                    ),
                ]
            ).astype(np.float32),
        }

        # extra observation for critics
        extra_dict = {
            "key1_pts": info["key1_pts_m"],
            "key2_pts": info["key2_pts_m"],
            "key_side_pts": info["key_side_pts_m"],
            "lock1_pts": info["lock1_pts_m"],
            "lock2_pts": info["lock2_pts_m"],
            "lock_side_pts": info["lock_side_pts_m"],
            "relative_motion": info["relative_motion_mm"].astype(np.float32),
        }
        obs_dict.update(extra_dict)

        return obs_dict

    def get_reward(self, info) -> float:
        """
        Calculates the reward based on the environment's state information.

        This function determines the reward for the current state by evaluating the progress towards solving the lock.
        It considers the reduction in error, the avoidance of large forces, and the success of the action.

        Parameters:
        - info (dict): A dictionary containing the environment's state information, obtained from `get_info`.

        Returns:
        - reward (float): The calculated reward for the current state and action.
        """
        self.error_evaluation_list.append(self.evaluate_error(info))
        reward = -self.step_penalty
        reward += self.error_evaluation_list[-2] - self.error_evaluation_list[-1]

        # punish large force
        surface_diff = info["surface_diff_m"].clip(0.2e-3, 1.5e-3) * 1000
        reward -= np.sum(surface_diff)

        if info["is_success"]:
            reward += self.final_reward
        elif info["tactile_movement_too_large"] or info["error_too_large"]:
            # prevent the agent from suicide
            reward += (
                -10
                * self.step_penalty
                * (self.max_steps - self.current_episode_elapsed_steps)
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
            info["steps"] >= self.max_steps
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
            self.tactile_sensor_1.get_surface_vertices_world(),
            self.tactile_sensor_2.get_surface_vertices_world(),
        ]

    def close(self):
        self.ipc_system = None
        pass


class LongOpenLockRandPointFlowEnv(LongOpenLockSimEnv):
    """
    An environment class that extends LongOpenLockSimEnv to incorporate random point flow for tactile sensors.

    This class adds functionality for handling random point flow observations from tactile sensors,
    which can be used to simulate realistic sensor noise and dynamics.

    Parameters:
    - marker_interval_range (Tuple[float, float]): A tuple representing the range of intervals between marker points in millimeters.
    - marker_rotation_range (float): The range of overall marker rotation in radians.
    - marker_translation_range (Tuple[float, float]): A tuple representing the range of overall marker translation in millimeters.
    - marker_pos_shift_range (Tuple[float, float]): A tuple representing the range of independent marker position shift in millimeters.
    - marker_random_noise (float): The standard deviation of Gaussian noise applied to marker points in pixels.
    - marker_lose_tracking_probability (float): The probability of losing tracking for each marker.
    - normalize (bool): A flag indicating whether to normalize the observations.
    - **kwargs: Additional keyword arguments passed to the parent class LongOpenLockSimEnv.

    The class is designed to provide a more realistic simulation environment by incorporating random point flow observations,
    which can be used for training and testing control policies under noisy conditions.
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
        Initializes a new instance of LongOpenLockRandPointFlowEnv.

        This method sets up the environment with the specified parameters for random point flow and
        calls the parent class's constructor to complete the initialization.

        Parameters:
        - marker_interval_range, marker_rotation_range, marker_translation_range, marker_pos_shift_range,
          marker_random_noise, marker_lose_tracking_probability, normalize: See class documentation for details.
        - **kwargs: See parent class LongOpenLockSimEnv for additional parameters.
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
        self.default_camera_params = np.array([0, 0, 0, 0, 0, 530, 530, 0, 2.4])
        self.marker_flow_size = 128

        super(LongOpenLockRandPointFlowEnv, self).__init__(**kwargs)

    def _get_sensor_surface_vertices(self) -> list[np.ndarray, np.ndarray]:
        """
        Retrieves the surface vertices of the tactile sensors in the camera coordinate system.

        This function is used to get the current surface vertices of the tactile sensors, which can be used to
        calculate the relative motion or deformation of the surface in the context of the camera view.

        Returns:
        - vertices (list of np.ndarray): A list containing the surface vertices of the left and right tactile sensors
        in the camera coordinate system.

        """
        return [
            self.tactile_sensor_1.get_surface_vertices_camera(),
            self.tactile_sensor_2.get_surface_vertices_camera(),
        ]

    def add_tactile_sensors(self, init_pos_l, init_rot_l, init_pos_r, init_rot_r):
        """
        Initializes and adds two tactile sensors, specifically VisionTactileSensorSapienIPC instances, to the simulation environment.

        This function creates two tactile sensors with the VisionTactileSensorSapienIPC class, one for each side (left and right)
        of the lock mechanism.
        Each sensor is configured with its own initial position, rotation, and additional parameters specific to the VisionTactileSensorSapienIPC.

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
        Constructs the observation dictionary for the environment, including random point flow data.

        This function generates the observation that an agent would receive after taking an action in the environment.
        In addition to the standard observations from the parent class, this function includes random point flow data from the tactile sensors.

        Parameters:
        - info (dict): Optional dictionary containing additional environment state information.

        Returns:
        - obs (dict): A dictionary containing the observation data, including random point flow from tactile sensors.

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

        key1_pts = obs.pop("key1_pts")
        key2_pts = obs.pop("key2_pts")
        obs["key1"] = (
            np.array(
                [
                    key1_pts.mean(0)[0] - info["lock1_pts_m"].mean(0)[0],
                    key1_pts.mean(0)[1],
                    key1_pts.mean(0)[2] - 0.03,
                ],
                dtype=np.float32,
            )
            * 200.0
        )
        obs["key2"] = (
            np.array(
                [
                    key2_pts.mean(0)[0] - info["lock2_pts_m"].mean(0)[0],
                    key2_pts.mean(0)[1],
                    key2_pts.mean(0)[2] - 0.03,
                ],
                dtype=np.float32,
            )
            * 200.0
        )

        return obs


if __name__ == "__main__":

    def visualize_marker_point_flow(o, i, name, save_dir="marker_flow_images_4"):

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
        filename = os.path.join(save_dir, f"sp-from-sapien-{name}-marker_flow_{i}.png")
        plt.savefig(filename)
        plt.close()

    use_gui = True
    use_render_rgb = True
    timestep = 0.05

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

    params = LongOpenLockParams(
        sim_time_step=timestep,
        tac_sensor_meta_file="gelsight_mini_e430/meta_file",
        key_lock_path_file="configs/key_and_lock/key_lock.txt",
        indentation_depth_mm=1.0,
        elastic_modulus_r=3e5,
        elastic_modulus_l=3e5,
        key_friction=1.0,
        lock_friction=1.0,
    )
    print(params)

    env = LongOpenLockRandPointFlowEnv(
        params=params,
        params_upper_bound=params,
        gui=use_gui,
        step_penalty=1,
        final_reward=10,
        max_action_mm=np.array([2, 2]),
        max_steps=80,
        key_x_max_offset_mm=0,
        key_y_max_offset_mm=0,
        key_z_max_offset_mm=0,
        sensor_offset_x_range_len_mm=2.0,
        sensor_offset_z_range_len_mm=2.0,
        render_rgb=use_render_rgb,
        marker_interval_range=(2.0625, 2.0625),
        marker_rotation_range=0.0,
        marker_translation_range=(0.0, 0.0),
        marker_pos_shift_range=(0.0, 0.0),
        marker_random_noise=0.1,
        marker_lose_tracking_probability=0.0,
        log_path=log_folder,
        logger=log,
        normalize=False,
        device="cuda:0",
        no_render=False,
        env_type="test",
    )

    np.set_printoptions(precision=4)

    offset = [0, 0, 0]
    o, _ = env.reset(key_idx=3)
    for k, v in o.items():
        test_log.info(f"{k} : {v.shape}")
    info = env.get_info()
    test_log.info(f"timestep: {timestep}")
    test_log.info(f"info : {info}\n")

    for i in range(7):
        obs, rew, done, _, info = env.step(np.array([-0.5, 0.0]))
        visualize_marker_point_flow(obs, i, "test")
        test_log.info(
            f"step: {env.current_episode_elapsed_steps:2d} rew: {rew:.2f} done: {done} success: {info['is_success']} re: {info['relative_motion_mm']}"
        )
    for i in range(10):
        obs, rew, done, _, info = env.step(np.array([0.0, -0.5]))
        visualize_marker_point_flow(obs, i + 24, "test")
        test_log.info(
            f"step: {env.current_episode_elapsed_steps:2d} rew: {rew:.2f} done: {done} success: {info['is_success']}"
        )
