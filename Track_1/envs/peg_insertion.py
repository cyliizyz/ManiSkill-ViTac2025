import copy
import json
import math
import os
import sys

from sapienipc.ipc_utils.user_utils import ipc_update_render_all

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
import time
from typing import Tuple, Union

import fcl
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import transforms3d as t3d
import warp as wp
from gymnasium import spaces
from path import Path
from sapien.utils.viewer import Viewer as viewer
from sapienipc.ipc_system import IPCSystem, IPCSystemConfig

from envs.common_params import CommonParams
from envs.tactile_sensor_sapienipc import (
    TactileSensorSapienIPC,
    VisionTactileSensorSapienIPC,
)
from utils.common import randomize_params, suppress_stdout_stderr
from utils.geometry import quat_product
from utils.gym_env_utils import convert_observation_to_space
from utils.sapienipc_utils import build_sapien_entity_ABD

wp.init()
wp_device = wp.get_preferred_device()

gui = False


def evaluate_error(offset):
    return np.linalg.norm(offset)


class ContinuousInsertionParams(CommonParams):
    def __init__(
        self,
        gripper_x_offset: float = 0.0,
        gripper_z_offset: float = 0.0,
        indentation_depth: float = 1.0,
        peg_friction: float = 1.0,
        hole_friction: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gripper_x_offset = gripper_x_offset
        self.gripper_z_offset = gripper_z_offset
        self.indentation_depth = indentation_depth
        self.peg_friction = peg_friction
        self.hole_friction = hole_friction


class ContinuousInsertionSimEnv(gym.Env):
    def __init__(
        self,
        step_penalty: float,
        final_reward: float,
        max_action: np.ndarray,
        max_steps: int = 15,
        z_step_size: float = 0.075,
        peg_hole_path_file: str = "",
        peg_x_max_offset: float = 5.0,
        peg_y_max_offset: float = 5.0,
        peg_theta_max_offset: float = 10.0,
        obs_check_threshold: float = 1e-3,
        params=None,
        params_upper_bound=None,
        device: str = "cuda:0",
        no_render: bool = False,
        **kwargs,
    ):
        """
        Initialize the ContinuousInsertionSimEnv.

        :param step_penalty: Penalty for each step taken in the environment.
        :param final_reward: Reward given when the task is successfully completed.
        assert max_action.shape == (3,), f"max_action should have shape (3,), but got shape {max_action.shape}"
        :param max_steps: Maximum number of steps allowed in an episode.
        :param z_step_size: Step size in the z-direction.
        :param peg_hole_path_file: Path to the file containing peg and hole paths.
        :param peg_x_max_offset: Maximum offset in the x-direction for the peg.
        :param peg_y_max_offset: Maximum offset in the y-direction for the peg.
        :param peg_theta_max_offset: Maximum offset in the theta direction for the peg.
        :param obs_check_threshold: Threshold for checking observations.
        :param params: Lower bound parameters for the environment.
        :param params_upper_bound: Upper bound parameters for the environment.
        :param device: Device to be used for simulation, default is "cuda:0".
        :param no_render: Flag to disable rendering.
        :param kwargs: Additional keyword arguments.
        """
        super(ContinuousInsertionSimEnv, self).__init__()

        self.no_render = no_render
        self.step_penalty = step_penalty
        self.final_reward = final_reward

        self.max_steps = max_steps
        self.z_step_size = z_step_size
        peg_hole_path_file = Path(track_path) / peg_hole_path_file
        self.peg_hole_path_list = []
        with open(peg_hole_path_file, "r") as f:
            for l in f.readlines():
                self.peg_hole_path_list.append(
                    [ss.strip() for ss in l.strip().split(",")]
                )
        self.peg_x_max_offset = peg_x_max_offset
        self.peg_y_max_offset = peg_y_max_offset
        self.peg_theta_max_offset = peg_theta_max_offset
        self.obs_check_threshold = obs_check_threshold

        if params is None:
            self.params_lb = ContinuousInsertionParams()
        else:
            self.params_lb = copy.deepcopy(params)

        if params_upper_bound is None:
            self.params_ub = copy.deepcopy(self.params_lb)
        else:
            self.params_ub = copy.deepcopy(params_upper_bound)

        self.params = randomize_params(
            self.params_lb, self.params_ub
        )  # type: ContinuousInsertionParams

        self.current_episode_elapsed_steps = 0
        self.current_episode_over = False
        self.sensor_grasp_center_init = None
        self.sensor_grasp_center_current = None
        # self.error_too_large = False
        # self.too_many_steps = False

        # build scene, system
        self.viewer = None

        if not no_render:
            self.scene = sapien.Scene()
            self.scene.set_ambient_light([1.0, 1.0, 1.0])
            self.scene.add_directional_light([0, -1, -1], [1.0, 1.0, 1.0], True)
        else:
            self.scene = sapien.Scene()

        # add a camera to indicate shader
        if not no_render:
            cam_entity = sapien.Entity()
            cam = sapien.render.RenderCameraComponent(512, 512)
            cam_entity.add_component(cam)
            cam_entity.name = "camera"
            self.scene.add_entity(cam_entity)

        ######## Create system ########
        ipc_system_config = IPCSystemConfig()
        # memory config
        ipc_system_config.max_scenes = 1
        ipc_system_config.max_surface_primitives_per_scene = 1 << 14
        ipc_system_config.max_blocks = 4000000
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

        # set device
        device = wp.get_device(device)
        ipc_system_config.device = wp.get_device(device)

        self.ipc_system = IPCSystem(ipc_system_config)
        self.scene.add_system(self.ipc_system)

        assert max_action.shape == (3,)
        self.max_action = max_action
        self.action_space = spaces.Box(
            low=-1, high=1, shape=max_action.shape, dtype=np.float32
        )
        self.default_observation = self.__get_sensor_default_observation__()
        self.observation_space = convert_observation_to_space(self.default_observation)

    def seed(self, seed=None):
        if seed is None:
            seed = (int(time.time() * 1000) % 10000 * os.getpid()) % 2**30
        np.random.seed(seed)

    def __get_sensor_default_observation__(self):

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

    def reset(self, offset=None, seed=None, peg_idx: int = None):

        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.params = randomize_params(self.params_lb, self.params_ub)
        self.current_episode_elapsed_steps = 0
        self.error_too_large = False
        self.too_many_steps = False

        if offset:
            offset = np.array(offset).astype(float)

        offset = self._initialize(offset, peg_idx)

        self.init_offset_of_current_episode = offset
        self.current_offset_of_current_episode = offset
        self.error_evaluation_list = []
        self.error_evaluation_list.append(
            evaluate_error(self.current_offset_of_current_episode)
        )
        self.current_episode_initial_left_surface_pts = self.no_contact_surface_mesh[0]
        self.current_episode_initial_right_surface_pts = self.no_contact_surface_mesh[1]
        self.current_episode_over = False

        return self.get_obs(), {}

    def _initialize(
        self, offset: Union[np.ndarray, None], peg_idx: Union[int, None] = None
    ):
        """
        offset: (x_offset in mm, y_offset in mm, theta_offset in degree)
        """

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
        print("Peg name:", peg_path)

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
        self.scene.add_entity(self.hole_entity)
        if self.hole_ext == ".msh":
            self.hole_upper_z = hole_height = np.max(
                hole_abd.tet_mesh.vertices[:, 2]
            ) - np.min(hole_abd.tet_mesh.vertices[:, 2])
        else:
            self.hole_upper_z = hole_height = np.max(
                hole_abd.tri_mesh.vertices[:, 2]
            ) - np.min(hole_abd.tri_mesh.vertices[:, 2])

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
            peg_width = np.max(peg_abd.tet_mesh.vertices[:, 1]) - np.min(
                peg_abd.tet_mesh.vertices[:, 1]
            )
            peg_height = np.max(peg_abd.tet_mesh.vertices[:, 2]) - np.min(
                peg_abd.tet_mesh.vertices[:, 2]
            )
            self.peg_bottom_pts_id = np.where(
                peg_abd.tet_mesh.vertices[:, 2]
                < np.min(peg_abd.tet_mesh.vertices[:, 2]) + 1e-4
            )[0]
        else:
            peg_width = np.max(peg_abd.tri_mesh.vertices[:, 1]) - np.min(
                peg_abd.tri_mesh.vertices[:, 1]
            )
            peg_height = np.max(peg_abd.tri_mesh.vertices[:, 2]) - np.min(
                peg_abd.tri_mesh.vertices[:, 2]
            )
            self.peg_bottom_pts_id = np.where(
                peg_abd.tri_mesh.vertices[:, 2]
                < np.min(peg_abd.tri_mesh.vertices[:, 2]) + 1e-4
            )[0]

        # generate random and valid offset
        if offset is None:
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
                x_offset = (np.random.rand() * 2 - 1) * self.peg_x_max_offset / 1000
                y_offset = (np.random.rand() * 2 - 1) * self.peg_y_max_offset / 1000
                theta_offset = (
                    (np.random.rand() * 2 - 1) * self.peg_theta_max_offset * np.pi / 180
                )

                R = t3d.euler.euler2mat(0.0, 0.0, theta_offset, axes="rxyz")
                T = np.array([x_offset, y_offset, 0.0])
                t3 = fcl.Transform(R, T)
                peg_fcl.setTransform(t3)

                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()

                ret = fcl.collide(peg_fcl, hole_fcl, request, result)

                if ret > 0:
                    offset = np.array(
                        [x_offset * 1000, y_offset * 1000, theta_offset * 180 / np.pi]
                    )
                    break
        else:
            x_offset, y_offset, theta_offset = (
                offset[0] / 1000,
                offset[1] / 1000,
                offset[2] * np.pi / 180,
            )

        # add peg to the scene
        init_pos = (
            x_offset,
            y_offset,
            hole_height + 0.1e-3,
        )
        init_theta_offset = theta_offset
        peg_offset_quat = t3d.quaternions.axangle2quat((0, 0, 1), theta_offset, True)
        self.peg_entity.set_pose(sapien.Pose(p=init_pos, q=peg_offset_quat))
        self.scene.add_entity(self.peg_entity)

        # add tactile sensors to the sapien scene
        gripper_x_offset = self.params.gripper_x_offset / 1000  # mm to m
        gripper_z_offset = self.params.gripper_z_offset / 1000
        sensor_grasp_center = np.array( 
            (
                math.cos(theta_offset) * gripper_x_offset + init_pos[0],
                math.sin(theta_offset) * gripper_x_offset + init_pos[1],
                peg_height + init_pos[2] + gripper_z_offset,
            )
        )
        init_pos_l = (
            -math.sin(theta_offset) * (peg_width / 2 + 0.0020 + 0.0001)
            + sensor_grasp_center[0],
            math.cos(theta_offset) * (peg_width / 2 + 0.0020 + 0.0001)
            + sensor_grasp_center[1],
            sensor_grasp_center[2],
        )
        init_pos_r = (
            math.sin(theta_offset) * (peg_width / 2 + 0.0020 + 0.0001)
            + sensor_grasp_center[0],
            -math.cos(theta_offset) * (peg_width / 2 + 0.0020 + 0.0001)
            + sensor_grasp_center[1],
            sensor_grasp_center[2],
        )
        init_rot_l = quat_product(peg_offset_quat, (0.5, 0.5, 0.5, -0.5))
        init_rot_r = quat_product(peg_offset_quat, (0.5, -0.5, 0.5, 0.5))
        with suppress_stdout_stderr():
            self._add_tactile_sensors(init_pos_l, init_rot_l, init_pos_r, init_rot_r)

        # get init sensor center
        sensor_grasp_center = tuple((x + y) / 2 for x, y in zip(init_pos_l, init_pos_r))
        self.sensor_grasp_center_init = (
            np.array(sensor_grasp_center + (init_theta_offset,)) * 1000
        )
        self.sensor_grasp_center_current = self.sensor_grasp_center_init.copy()

        # gui initialization
        if gui:
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
                (0.1 + self.params.indentation_depth)
                / 1000
                / 5e-3
                / (self.params.sim_time_step if self.params.sim_time_step != 0 else 1)
            ),
            1,
        )
        grasp_speed = (
            (0.1 + self.params.indentation_depth)
            / 1000
            / grasp_step
            / self.params.sim_time_step
        )
        for _ in range(grasp_step):
            self.tactile_sensor_1.set_active_v(
                [
                    grasp_speed * math.sin(theta_offset),
                    -grasp_speed * math.cos(theta_offset),
                    0,
                ]
            )
            self.tactile_sensor_2.set_active_v(
                [
                    -grasp_speed * math.sin(theta_offset),
                    grasp_speed * math.cos(theta_offset),
                    0,
                ]
            )
            self.hole_abd.set_kinematic_target(
                np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
            )  # hole stays static
            self.ipc_system.step()
            self.tactile_sensor_1.step()
            self.tactile_sensor_2.step()
            if gui:
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
        z_distance = 0.1e-3 + self.z_step_size * 1e-3
        pre_insertion_step = max(
            round((z_distance / 1e-3) / self.params.sim_time_step), 1
        )
        pre_insertion_speed = (
            z_distance / pre_insertion_step / self.params.sim_time_step
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
            if gui:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

        return offset

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
        )

    def step(self, action):
        """
        :param action: numpy array; action[0]: delta_x, mm; action[1]: delta_y, mm; action[2]: delta_theta, radian.
        :return: Tuple[dict, float, bool, bool, dict]
        """
        self.current_episode_elapsed_steps += 1
        action = np.array(action).flatten() * self.max_action
        self._sim_step(action)

        info = self.get_info()
        obs = self.get_obs(info=info)
        reward = self.get_reward(info=info, obs=obs)
        terminated = self.get_terminated(info=info, obs=obs)
        truncated = self.get_truncated(info=info, obs=obs)
        return obs, reward, terminated, truncated, info

    def _sim_step(self, action):
        action = np.clip(action, -self.max_action, self.max_action)
        current_theta = self.current_offset_of_current_episode[2] * np.pi / 180
        action_x = action[0] * math.cos(current_theta) - action[1] * math.sin(
            current_theta
        )
        action_y = action[0] * math.sin(current_theta) + action[1] * math.cos(
            current_theta
        )
        action_z = -self.z_step_size

        action_theta = action[2]
        action_theta_degree = action[2] * np.pi / 180

        self.current_offset_of_current_episode[0] += action_x
        self.current_offset_of_current_episode[1] += action_y
        self.current_offset_of_current_episode[2] += action_theta
        self.sensor_grasp_center_current[0] += action_x
        self.sensor_grasp_center_current[1] += action_y
        self.sensor_grasp_center_current[2] += action_z
        self.sensor_grasp_center_current[3] += action_theta_degree

        action_sim = np.array([action_x, action_y, action_theta])
        sensor_grasp_center = (
            self.tactile_sensor_1.current_pos + self.tactile_sensor_2.current_pos
        ) / 2

        if (
            abs(self.current_offset_of_current_episode[0]) > 12 + 1e-5
            or abs(self.current_offset_of_current_episode[1]) > 12 + 1e-5
            or (abs(self.current_offset_of_current_episode[2]) > 15 + 1e-5)
        ):

            self.error_too_large = (
                True  # if error is loo large, then no need to do simulation
            )
        elif self.current_episode_elapsed_steps > self.max_steps:
            self.too_many_steps = True  # This condition is unlikely because the environment should terminate before reaching this point
        else:
            x = action_sim[0] / 1000
            y = action_sim[1] / 1000
            theta = action_sim[2] * np.pi / 180

            action_substeps = max(
                1, round((max(abs(x), abs(y)) / 5e-3) / self.params.sim_time_step)
            )
            action_substeps = max(
                action_substeps, round((abs(theta) / 0.2) / self.params.sim_time_step)
            )
            v_x = x / self.params.sim_time_step / action_substeps
            v_y = y / self.params.sim_time_step / action_substeps
            v_theta = theta / self.params.sim_time_step / action_substeps

            for _ in range(action_substeps):
                self.tactile_sensor_1.set_active_v_r(
                    [v_x, v_y, 0],
                    sensor_grasp_center,
                    (0, 0, 1),
                    v_theta,
                )
                self.tactile_sensor_2.set_active_v_r(
                    [v_x, v_y, 0],
                    sensor_grasp_center,
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
                sensor_grasp_center = (
                    self.tactile_sensor_1.current_pos
                    + self.tactile_sensor_2.current_pos
                ) / 2
                if (not state1) or (not state2):
                    self.error_too_large = True
                if gui:
                    self.scene.update_render()
                    ipc_update_render_all(self.scene)
                    self.viewer.render()
        z = -self.z_step_size / 1000
        z_substeps = max(1, round(abs(z) / 5e-3 / self.params.sim_time_step))
        v_z = z / self.params.sim_time_step / z_substeps
        for _ in range(z_substeps):
            self.tactile_sensor_1.set_active_v(
                [0, 0, v_z],
            )
            self.tactile_sensor_2.set_active_v(
                [0, 0, v_z],
            )
            # with suppress_stdout_stderr():
            self.hole_abd.set_kinematic_target(
                np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0)
            )  # hole stays static
            self.ipc_system.step()
            state1 = self.tactile_sensor_1.step()
            state2 = self.tactile_sensor_2.step()
            if (not state1) or (not state2):
                self.error_too_large = True
            if gui:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

    def get_info(self):
        info = {"steps": self.current_episode_elapsed_steps}

        peg_relative_z = self._get_peg_relative_z()
        info["peg_relative_z"] = peg_relative_z
        info["is_success"] = False
        info["error_too_large"] = False
        info["too_many_steps"] = False
        info["observation_check"] = (-1.0, -1.0)

        if self.error_too_large:
            info["error_too_large"] = True
        elif self.too_many_steps:
            info["too_many_steps"] = True
        elif (
            self.current_episode_elapsed_steps * self.z_step_size > 0.35
            and np.sum(peg_relative_z < -0.3e-3) == peg_relative_z.shape[0]
        ):
            observation_left_surface_pts, observation_right_surface_pts = (
                self._get_sensor_surface_vertices()
            )
            l_diff = np.mean(
                np.sqrt(
                    np.sum(
                        (
                            self.current_episode_initial_left_surface_pts
                            - observation_left_surface_pts
                        )
                        ** 2,
                        axis=-1,
                    )
                )
            )
            r_diff = np.mean(
                np.sqrt(
                    np.sum(
                        (
                            self.current_episode_initial_right_surface_pts
                            - observation_right_surface_pts
                        )
                        ** 2,
                        axis=-1,
                    )
                )
            )
            if l_diff < self.obs_check_threshold and r_diff < self.obs_check_threshold:
                info["is_success"] = True
                info["observation_check"] = (l_diff, r_diff)
            else:
                info["observation_check"] = (l_diff, r_diff)

        return info

    def get_obs(self, info=None):
        sensor_offset = self.sensor_grasp_center_current - self.sensor_grasp_center_init

        if info:
            if info["error_too_large"] or info["too_many_steps"]:
                obs_dict = {
                    "surface_pts": np.stack(
                        [
                            np.stack(
                                [self.current_episode_initial_left_surface_pts] * 2
                            ),
                            np.stack(
                                [self.current_episode_initial_right_surface_pts] * 2
                            ),
                        ]
                    ).astype(np.float32),
                    "gt_offset": np.array(self.current_offset_of_current_episode, dtype=np.float32),
                    "relative_motion": np.array(sensor_offset, dtype=np.float32)

                }
                return obs_dict

        observation_left_surface_pts, observation_right_surface_pts = (
            self._get_sensor_surface_vertices()
        )

        obs_dict = {
            "surface_pts": np.stack(
                [
                    np.stack(
                        [
                            self.current_episode_initial_left_surface_pts,
                            observation_left_surface_pts,
                        ]
                    ),
                    np.stack(
                        [
                            self.current_episode_initial_right_surface_pts,
                            observation_right_surface_pts,
                        ]
                    ),
                ]
            ).astype(np.float32),
            "gt_offset": np.array(self.current_offset_of_current_episode, dtype=np.float32),
            "relative_motion": np.array(sensor_offset, dtype=np.float32)
        }

        return obs_dict

    def get_reward(self, info, obs=None):
        self.error_evaluation_list.append(
            evaluate_error(self.current_offset_of_current_episode)
        )
        reward = (
            self.error_evaluation_list[-2]
            - self.error_evaluation_list[-1]
            - self.step_penalty
        )

        if info["too_many_steps"]:
            reward = 0
        elif info["error_too_large"]:
            reward += (
                -2
                * self.step_penalty
                * (self.max_steps - self.current_episode_elapsed_steps)
                + self.step_penalty
            )
        elif info["is_success"]:
            reward += self.final_reward

        return reward

    def get_truncated(self, info, obs=None):
        return info["steps"] >= self.max_steps

    def get_terminated(self, info, obs=None):
        return info["error_too_large"] or info["is_success"]

    def _get_sensor_surface_vertices(self):
        return [
            self.tactile_sensor_1.get_surface_vertices_sensor(),
            self.tactile_sensor_2.get_surface_vertices_sensor(),
        ]

    def _get_peg_relative_z(self):
        peg_pts = self.peg_abd.get_positions().cpu().numpy().copy()
        peg_bottom_z = peg_pts[self.peg_bottom_pts_id][:, 2]
        # print(peg_bottom_z)
        return peg_bottom_z - self.hole_upper_z

    def close(self):
        self.ipc_system = None
        pass


class ContinuousInsertionSimGymRandomizedPointFLowEnv(ContinuousInsertionSimEnv):
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
        Initialize the ContinuousInsertionSimGymRandomizedPointFLowEnv.

        Parameters:
        marker_interval_range (Tuple[float, float]): Range of intervals between markers in mm.
        marker_rotation_range (float): Overall marker rotation range in radians.
        marker_translation_range (Tuple[float, float]): Overall marker translation range in mm.
                                                        First two elements for x-axis, last two elements for y-axis.
        marker_pos_shift_range (Tuple[float, float]): Independent marker position shift range in mm,
                                                      in x- and y-axis, caused by fabrication errors.
        marker_random_noise (float): Standard deviation of Gaussian marker noise in pixels,
                                     caused by CMOS noise and image processing.
        marker_lose_tracking_probability (float): Probability of losing tracking, applied to each marker.
        normalize (bool): Whether to normalize the marker flow.
        kwargs: Additional keyword arguments for the parent class.
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

        super(ContinuousInsertionSimGymRandomizedPointFLowEnv, self).__init__(**kwargs)

        self.default_observation = {
            "relative_motion": np.zeros((4,), dtype=np.float32),
            "gt_offset": np.zeros((3,), dtype=np.float32),
            "marker_flow": np.zeros((2, 2, self.marker_flow_size, 2), dtype=np.float32),
        }

        self.observation_space = convert_observation_to_space(self.default_observation)

    def _get_sensor_surface_vertices(self):
        return [
            self.tactile_sensor_1.get_surface_vertices_camera(),
            self.tactile_sensor_2.get_surface_vertices_camera(),
        ]

    def _add_tactile_sensors(self, init_pos_l, init_rot_l, init_pos_r, init_rot_r):
        self.tactile_sensor_1 = VisionTactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_l,
            init_rot=init_rot_l,
            elastic_modulus=self.params.tac_elastic_modulus_l,
            poisson_ratio=self.params.tac_poisson_ratio_l,
            density=self.params.tac_density_l,
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
        )

    def get_obs(self, info=None):
        obs = super().get_obs(info=info)
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
    gui = True
    timestep = 0.05

    params = ContinuousInsertionParams(
        sim_time_step=0.1,
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
        tac_elastic_modulus_l=3.0e5,  # note if 3e5 is correctly recognized as float
        tac_poisson_ratio_l=0.3,
        tac_density_l=1e3,
        tac_elastic_modulus_r=3.0e5,
        tac_poisson_ratio_r=0.3,
        tac_density_r=1e3,
        tac_friction=100,
        # task specific parameters
        gripper_x_offset=0,
        gripper_z_offset=-4,
        indentation_depth=1,
        peg_friction=10,
        hole_friction=1,
    )
    print(params)

    env = ContinuousInsertionSimGymRandomizedPointFLowEnv(
        params=params,
        step_penalty=1,
        final_reward=10,
        max_action=np.array([2, 2, 4]),
        max_steps=10,
        z_step_size=0.5,
        marker_interval_range=(2.0625, 2.0625),
        marker_rotation_range=0.0,
        marker_translation_range=(0.0, 0.0),
        marker_pos_shift_range=(0.0, 0.0),
        marker_random_noise=0.1,
        normalize=False,
        peg_hole_path_file="configs/peg_insertion/3shape_1.5mm.txt",
    )

    np.set_printoptions(precision=4)

    def visualize_marker_point_flow(o, i, name, save_dir="marker_flow_images3"):

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

    offset_list = [[4, 0, 0], [-4, 0, 0], [0, 4, 0], [0, -4, 0]]
    for offset in offset_list:
        # offset = [4,0,0]
        o, _ = env.reset(offset)
        for k, v in o.items():
            print(k, v.shape)

        for i in range(10):
            action = [-0.2, 0, 0]
            o, r, d, _, info = env.step(action)
            print(
                f"step: {info['steps']} reward: {r:.2f} gt_offset: {o['gt_offset']} success: {info['is_success']}  relative_motion: {o['relative_motion']}"
                f" peg_z: {info['peg_relative_z']}, obs check: {info['observation_check']}")
            # visualize_marker_point_flow(o, i, str(offset), save_dir="saved_images")
        if env.viewer is not None:
            while True:
                if env.viewer.window.key_down("c"):
                    break
                env.scene.update_render()
                ipc_update_render_all(env.scene)
                env.viewer.render()
