#! /usr/bin/python
import math
import os
import pickle
import sys
import time

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)
sys.path.append(os.path.join(script_path, ".."))

import gym
import numpy as np
import rospy

from utils.RL_common_utils import get_marker_flow_rotation_and_force, visualize_marker_point_flow

from StageController.combined_stage import CombinedStage
from cv_bridge import CvBridge
from gym import spaces
from gelsight_mini_ros.msg import judging_msg, tracking_msg
from gelsight_mini_ros.srv import ResetMarkerTracker

from motion_manager_stage import MotionManagerStage
from utils.RL_common_utils import get_rotation_and_force, manual_policy, evaluate_error
from utils.np_utils import generate_offset
from utils.data_process_utils import check_blocked, normalize_tactile_flow_map
from utils.utils import ThreadSafeContainer
from utils.data_process_utils import adapt_marker_seq_to_unified_size

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)

bridge = CvBridge()

left_marker_flow_container = ThreadSafeContainer(max_size=30)
right_marker_flow_container = ThreadSafeContainer(max_size=30)
is_movement = False
is_contact = False


def callback_sensor(data: judging_msg):
    global is_contact
    global is_movement
    contact_msg = data
    is_movement = contact_msg.is_overforced
    is_contact = contact_msg.is_contact


def callback_marker_flow_left(data: tracking_msg):
    global left_marker_flow_container
    marker_init_pos = np.stack([data.marker_x, data.marker_y]).transpose()
    marker_displacement = np.stack([data.marker_displacement_x, data.marker_displacement_y]).transpose()
    marker_cur_pos = marker_init_pos + marker_displacement
    marker_observation = np.stack([marker_init_pos, marker_cur_pos])
    left_marker_flow_container.put(marker_observation)


def callback_marker_flow_right(data: tracking_msg):
    global right_marker_flow_container
    marker_init_pos = np.stack([data.marker_x, data.marker_y]).transpose()
    marker_displacement = np.stack([data.marker_displacement_x, data.marker_displacement_y]).transpose()
    marker_cur_pos = marker_init_pos + marker_displacement
    marker_observation = np.stack([marker_init_pos, marker_cur_pos])
    right_marker_flow_container.put(marker_observation)


class ContinuousInsertionRealPointFlowEnvironment(gym.Env):
    def __init__(self, motion_manager, max_error, penalty, final_reward,
                 max_action, max_steps=15, z_step_size=0.075, peg='cuboid', clearance=2.5, normalize=False,
                 grasp_height_offset=0):
        # environment parameters
        self.max_error = np.array(max_error)
        self.max_action = np.array(max_action)
        self.step_penalty = penalty
        self.final_reward = final_reward
        self.max_steps = max_steps
        self.z_step_size = z_step_size
        self.current_episode_elapsed_steps = 0
        self.error_too_large = False
        self.too_many_steps = False
        self.tactile_movement_too_large = False
        self.current_episode_max_tactile_diff = 0
        self.current_episode_over = False
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "marker_flow": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 2, 128, 2), dtype=np.float32),
            "gt_offset": spaces.Box(low=np.array([-10, -10, -20]), high=np.array([10, 10, 20]), shape=(3,),
                                    dtype=np.float32)
        })
        self.peg = peg
        self.clearance = clearance
        self.normalize_flow = normalize
        self.grasp_height_offset = grasp_height_offset

        # ROS communication
        rospy.init_node('continuous_insertion_environment', anonymous=True)
        left_sensor_topic_name = "Marker_Tracking_Left"
        right_sensor_topic_name = "Marker_Tracking_Right"

        self.sub_tactile_fb = rospy.Subscriber("Marker_Tracking_Contact", judging_msg,
                                               callback=callback_sensor, callback_args=None,
                                               queue_size=10)
        self.sub_tactile_marker_flow_L = rospy.Subscriber(left_sensor_topic_name, tracking_msg,
                                                          callback=callback_marker_flow_left,
                                                          callback_args=None, queue_size=10)
        self.sub_tactile_marker_flow_R = rospy.Subscriber(right_sensor_topic_name, tracking_msg,
                                                          callback=callback_marker_flow_right,
                                                          callback_args=None, queue_size=10)

        self.left_sensor_init_marker_tracker_call = rospy.ServiceProxy("Marker_Tracking_Srv_Left", ResetMarkerTracker)
        self.right_sensor_init_marker_tracker_call = rospy.ServiceProxy("Marker_Tracking_Srv_Right", ResetMarkerTracker)
        global left_marker_flow_container
        global right_marker_flow_container
        global is_contact

        # motion stage related
        self.motion_manager = motion_manager
        self.motion_manager.set_grasp_height_offset(self.grasp_height_offset)
        self.motion_manager.set_peg(peg)
        self.motion_manager.set_clearance(clearance)
        self.motion_manager.go_to_safe_height()

        self.peg_in_gripper = False
        if not self.motion_manager.gripper.is_active():
            self.motion_manager.reset_gripper()
        else:
            # determine whether peg is in hand
            print("Gripper is already active")
            if self.motion_manager.get_gripper_pos() > 100:
                print(f"Gripper position is {self.motion_manager.get_gripper_pos()}")
                if is_contact:
                    self.peg_in_gripper = True
                    print(f"Peg is in gripper")
        self.reset_peg_pose()
        self.reset_marker_tracker()
        # get noise level, as a threshold
        left_marker_flow_container.clear()
        right_marker_flow_container.clear()
        while left_marker_flow_container.current_size < 30 or right_marker_flow_container.current_size < 30:
            time.sleep(0.1)

        # get marker noise level to determine the marker movement threshold
        left_marker_seq = left_marker_flow_container.get([i for i in range(30)], numpy=False)
        # (30, 2, marker_num, 2)
        right_marker_seq = right_marker_flow_container.get([i for i in range(30)], numpy=False)
        self.left_change_threshold = np.mean(
            [np.mean(
                np.sqrt(
                    np.sum(
                        (left_marker_seq[i][1, ...] - left_marker_seq[i][0, ...]) ** 2, axis=-1
                    ))) for i in range(30)]) * 1.25
        self.right_change_threshold = np.mean(
            [np.mean(
                np.sqrt(
                    np.sum(
                        (right_marker_seq[i][1, ...] - right_marker_seq[i][0, ...]) ** 2, axis=-1
                    ))) for i in range(30)]) * 1.25
        if self.left_change_threshold < 0.15:
            self.left_change_threshold = 0.15
        if self.right_change_threshold < 0.15:
            self.right_change_threshold = 0.15
        print(
            f"Tactile marker displacement threshold: {self.left_change_threshold:.2f}, {self.right_change_threshold:.2f}")

    """
    motion related methods
    """

    def reset_marker_tracker(self):
        if not (self.left_sensor_init_marker_tracker_call() and self.right_sensor_init_marker_tracker_call()):
            raise Exception("Init Marker Tracker Call Failed.")
        time.sleep(0.5)

    def reset_peg_pose(self):
        if self.peg_in_gripper:
            self.motion_manager.regrasp_peg()
        else:
            self.motion_manager.grasp_peg_from_garage()
        self.make_sure_peg_in_gripper()
        self.peg_in_gripper = True
        self.motion_manager.go_to_hole_origin()
        self.reset_marker_tracker()
        # time.sleep(0.5)

    def make_sure_peg_in_gripper(self):
        global is_contact
        time.sleep(1)
        if is_contact:
            self.peg_in_gripper = True
        else:
            self.peg_in_gripper = False
        while not self.peg_in_gripper:
            self.motion_manager.gripper.open()
            print("Peg is not in gripper. Now gripper is open.")
            print("Manual intervention is required.")
            time.sleep(10)
            self.motion_manager.grasp_peg_from_garage()
            if is_contact:
                self.peg_in_gripper = True
            else:
                self.peg_in_gripper = False

    @staticmethod
    def get_marker_flow():
        global left_marker_flow_container
        global right_marker_flow_container
        left_marker_flow = left_marker_flow_container.get(-1)
        right_marker_flow = right_marker_flow_container.get(-1)
        return left_marker_flow, right_marker_flow

    @staticmethod
    def get_marker_flow_difference(left_marker_flow, right_marker_flow):
        l_diff = np.mean(np.sqrt(np.sum((left_marker_flow[1, ...] - left_marker_flow[0, ...]) ** 2, axis=-1)))
        r_diff = np.mean(np.sqrt(np.sum((right_marker_flow[1, ...] - right_marker_flow[0, ...]) ** 2, axis=-1)))
        return l_diff, r_diff

    """
    RL env methods
    """

    def reset(self, specify_offset=None):
        # go to the original height
        self.current_episode_elapsed_steps = 0
        self.error_too_large = False
        self.too_many_steps = False
        self.tactile_movement_too_large = False
        self.current_episode_max_tactile_diff = 0

        print("Resetting...")
        self.motion_manager.go_to_lateral_move_height()
        time.sleep(0.5)

        l_diff, r_diff = self.get_marker_flow_difference(*self.get_marker_flow())
        print(f"Marker difference after last episode: {l_diff}, {r_diff}")
        if l_diff > self.left_change_threshold * 2.5 or r_diff > self.right_change_threshold * 2.5:
            print("Peg pose changed during insertion. Resetting...")
            self.reset_peg_pose()
            print("Resetting finished")

        self.reset_marker_tracker()
        blocked = False

        while not blocked:
            # if self.mono:
            #     offset = generate_mono_offset(5, 2.5)  # self.max_xy_error, 2.5, self.max_theta_error)
            # else:
            if specify_offset:
                offset = np.array(specify_offset).astype(float)
            else:
                offset = generate_offset(self.max_error[0], 0, self.max_error[2])
                offset[2] *= 180 / np.pi
            print("Initial offset is ", offset)
            offset = np.array(offset)
            offset_rad = offset.copy()
            offset_rad[2] /= 180 / np.pi
            if not check_blocked(offset_rad):
                specify_offset = None
                continue

            self.motion_manager.go_to_insertion_start_height()
            self.motion_manager.go_to_offset(offset[0], offset[1], offset[2])

            while left_marker_flow_container.current_size == 0 or right_marker_flow_container.current_size == 0:
                time.sleep(0.1)
            self.reset_marker_tracker()

            self.current_episode_initial_left_observation = adapt_marker_seq_to_unified_size(
                left_marker_flow_container.get(-1), 128)
            self.current_episode_initial_right_observation = adapt_marker_seq_to_unified_size(
                right_marker_flow_container.get(-1), 128)
            if self.normalize_flow:
                self.current_episode_initial_left_observation = self.current_episode_initial_left_observation / 160.0 - 1.0
                self.current_episode_initial_right_observation = self.current_episode_initial_right_observation / 160.0 - 1.0
            max_time = 2 / 0.4 * 0.95

            self.motion_manager.rel_move('z', -2, 0.4, wait=False)
            start_time = time.time()

            print("Timeout is " + str(max_time))
            blocked = True
            left_change, right_change = False, False
            while not (left_change or right_change):
                time.sleep(0.01)
                l_diff, r_diff = self.get_marker_flow_difference(*self.get_marker_flow())
                # print(f"Tactile image difference: {l_diff:.2f}, {r_diff:.2f}")
                if l_diff > self.left_change_threshold * 1.25:
                    left_change = True
                if r_diff > self.right_change_threshold * 1.25:
                    right_change = True
                if time.time() - start_time > max_time * 0.9:
                    blocked = False
                    break

            self.motion_manager.stop()
            self.motion_manager.wait_for_move_stop()

            if not blocked:
                print("Insertion succeeded at the beginning. Re-generate initial offset.")
                specify_offset = None
                continue
            else:
                self.init_offset_of_current_eposide = offset
                self.current_offset_of_current_episode = offset
                self.error_evaluation_list = []
                self.error_evaluation_list.append(evaluate_error(self.current_offset_of_current_episode))
                self.current_episode_initial_z = self.motion_manager.get_position()[2]
                self.motion_manager.rel_move('z', -self.z_step_size, vel=3, wait=True)
                self.motion_manager.wait_for_move_stop()
                time.sleep(0.1)

                self.current_episode_over = False
                self.obs_marker_flow = self.get_marker_flow()
                return self.get_obs()

    def _real_step(self, action):
        """
        :param action: numpy array; action[0]: delta_x, mm; action[1]: delta_y, mm; action[2]: delta_theta, radian.

        :return: observation, reward, done
        """
        action = np.clip(action, -self.max_action, self.max_action)

        # convert action to world coordinate
        current_theta = self.current_offset_of_current_episode[2] * np.pi / 180
        action_transformed_x = action[0] * math.cos(current_theta) - action[1] * math.sin(current_theta)
        action_transformed_y = action[0] * math.sin(current_theta) + action[1] * math.cos(current_theta)

        self.current_offset_of_current_episode[0] += action_transformed_x
        self.current_offset_of_current_episode[1] += action_transformed_y
        self.current_offset_of_current_episode[2] += action[2]

        global is_movement
        if self.current_episode_over:
            return None

        if self.current_offset_of_current_episode[0] ** 2 + self.current_offset_of_current_episode[1] ** 2 > 12 ** 2 \
                or (abs(self.current_offset_of_current_episode[2]) > 15):
            self.error_too_large = True  # if error is loo large, then no need to do real insertion
        elif self.current_episode_elapsed_steps > self.max_steps:
            self.too_many_steps = True  # normally not possible, because the env is already done at last step
        elif is_movement:
            self.tactile_movement_too_large = True
        else:
            self.motion_manager.go_to_offset(self.current_offset_of_current_episode[0],
                                             self.current_offset_of_current_episode[1],
                                             self.current_offset_of_current_episode[2])
            self.motion_manager.rel_move('z', -self.z_step_size, vel=3, wait=True)
            time.sleep(0.1)
            self.obs_marker_flow = self.get_marker_flow()

    def _success_check(self, z_distance=1):
        current_z = self.motion_manager.get_position()[2]
        left_marker_flow, right_marker_flow = self.get_marker_flow()
        l_diff_before_check, r_diff_before_check = self.get_marker_flow_difference(left_marker_flow, right_marker_flow)
        self.motion_manager.rel_move('z', -z_distance, vel=1.5, wait=False)  # check whether it is really blocked
        double_check_ok = True
        while self.motion_manager.is_moving():
            left_marker_flow, right_marker_flow = self.get_marker_flow()
            l_diff, r_diff = self.get_marker_flow_difference(left_marker_flow, right_marker_flow)
            print(f"Tactile difference: {l_diff:.2f}, {r_diff:.2f}")
            if not (l_diff < l_diff_before_check * 5 and r_diff < r_diff_before_check * 5):
                double_check_ok = False
                self.motion_manager.stop()
                self.motion_manager.wait_for_move_stop()
                self.obs_marker_flow = (left_marker_flow, right_marker_flow)
        if not double_check_ok:
            self.motion_manager.abs_move('z', current_z, vel=3, wait=True)
            self.motion_manager.wait_for_move_stop()
        return double_check_ok

    def step(self, action):
        self.current_episode_elapsed_steps += 1
        action = np.array(action).flatten() * self.max_action

        self._real_step(action)

        info = self.get_info()
        obs = self.get_obs(info=info)
        reward = self.get_reward(info=info, obs=obs)
        done = self.get_done(info=info, obs=obs)
        return obs, reward, done, info

    def get_info(self):
        info = {"steps": self.current_episode_elapsed_steps}
        info["is_success"] = False
        info["error_too_large"] = False
        info["too_many_steps"] = False
        info["tactile_movement_too_large"] = False
        info["message"] = "Normal step"
        current_z = self.motion_manager.get_position()[2]
        insertion_depth = self.current_episode_initial_z - current_z
        print(f"Insertion depth: {insertion_depth:.3f}")
        if self.error_too_large:
            info["error_too_large"] = True
            info["message"] = "Error too large, insertion attempt failed"
        elif self.too_many_steps:
            info["too_many_steps"] = True
            info["message"] = "Too many steps, insertion attempt failed"
        elif self.tactile_movement_too_large:
            info["tactile_movement_too_large"] = True
            info["message"] = "Tactile movement too large, insertion attempt failed"
        else:
            left_marker_flow, right_marker_flow = self.get_marker_flow()
            l_diff, r_diff = self.get_marker_flow_difference(left_marker_flow, right_marker_flow)
            print(f"Tactile difference: {l_diff:.2f}, {r_diff:.2f}")
            self.current_episode_max_tactile_diff = max(self.current_episode_max_tactile_diff, l_diff, r_diff)
            if insertion_depth > 0.35:
                relax_ratio = max(self.current_episode_max_tactile_diff / 1.5, 1)
                if l_diff < self.left_change_threshold * relax_ratio * 3 and r_diff < self.right_change_threshold * relax_ratio * 3:
                    double_check_success = self._success_check(z_distance=3)
                    if double_check_success:
                        info["is_success"] = True
                        info["message"] = "Insertion succeed！"
        return info

    def get_obs(self, info=None):
        if info:
            if info["error_too_large"] or info["too_many_steps"] or info["tactile_movement_too_large"]:
                obs_dict = {
                    "marker_flow": np.stack([self.current_episode_initial_left_observation,
                                             self.current_episode_initial_right_observation]).astype(np.float32),
                    "gt_offset": np.array(self.current_offset_of_current_episode, dtype=np.float32),
                }
                return obs_dict
        l_flow, r_flow = self.obs_marker_flow

        l_flow = adapt_marker_seq_to_unified_size(l_flow, 128)
        r_flow = adapt_marker_seq_to_unified_size(r_flow, 128)
        if self.normalize_flow:
            l_flow = l_flow / 160.0 - 1.0
            r_flow = r_flow / 160.0 - 1.0

        obs_dict = {
            "gt_offset": np.array(self.current_offset_of_current_episode, dtype=np.float32),
        }
        obs_dict.update({"marker_flow":
                             np.stack([l_flow, r_flow]).astype(np.float32)})

        return obs_dict

    def get_reward(self, info, obs=None):
        self.error_evaluation_list.append(evaluate_error(self.current_offset_of_current_episode))
        reward = self.error_evaluation_list[-2] - self.error_evaluation_list[-1] - self.step_penalty
        if info["too_many_steps"]:
            reward = 0
        elif info["error_too_large"] or info["tactile_movement_too_large"]:
            reward += -2 * self.step_penalty * (self.max_steps - self.current_episode_elapsed_steps) + self.step_penalty
        elif info["is_success"]:
            reward += self.final_reward

        return reward

    def get_done(self, info, obs=None):
        return info["is_success"] or info["steps"] >= self.max_steps or info["error_too_large"] or info[
            "tactile_movement_too_large"]

    def close(self):
        if self.peg_in_gripper:
            self.motion_manager.return_peg_to_garage()
            self.peg_in_gripper = False
        self.reset_marker_tracker()
        self.motion_manager.close()


if __name__ == "__main__":
    max_error = np.array([5, 5, 10])
    max_action = np.array([2, 2, 4])

    motion_manager = MotionManagerStage("/dev/translation_stage", '/dev/rotation_stage', '/dev/hande')
    env = ContinuousInsertionRealPointFlowEnvironment(motion_manager, max_error=max_error, penalty=2, final_reward=20,
                                                      max_steps=4,
                                                      max_action=max_action, z_step_size=0.125)
    # env.init_stage_and_gripper("/dev/translation_stage", '/dev/rotation_stage', '/dev/hande')

    succeed_num = 0
    offset_list = [(4, 0, 0), (0, 4, 0), (-4, 0, 0), (0, -4, 0)]
    exp_num = len(offset_list)
    total_steps = 0

    marker_flow_list = []
    for i in range(exp_num):
        print(f"\nEPISODE {i}.\n")
        obs = env.reset(offset_list[i])
        f_l = get_marker_flow_rotation_and_force(obs['marker_flow'][0])
        f_r = get_marker_flow_rotation_and_force(obs['marker_flow'][1])
        # f_x, f_z, M_x, M_y, M_z = get_total_force(f_l, f_r)
        next_action = [0, 0, 0]
        # next_action = manual_policy(f_l, f_r)
        next_action = np.array(next_action) / max_action
        done = False
        step = 0
        while not done:
            step += 1
            visualize_marker_point_flow(obs)
            print("--forces-- left:{:.2f}, {:.2f}, {:.2f}; right:{:.2f}, {:.2f}, {:.2f}".format(*f_l, *f_r))
            print(f"action: {next_action[0]:.2f}, {next_action[1]:.2f}, {next_action[2]:.2f}")
            ret = env.step(next_action)
            if ret:
                obs, reward, done, info = ret
                print(f"\nStep {step}.", info)
                # print(f"offset: {offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}")
                # next_action = np.random.randn(3)
                if not done:
                    f_l = get_marker_flow_rotation_and_force(obs['marker_flow'][0])
                    f_r = get_marker_flow_rotation_and_force(obs['marker_flow'][1])
                    next_action = [0, 0, 0]  # manual_policy(f_l, f_r)
                    next_action = np.array(next_action) / max_action
        marker_flow_list.append((offset_list[i], obs['marker_flow']))
        if info['is_success']:
            succeed_num += 1
            total_steps += step

    average_step = total_steps / max(succeed_num, 1)
    pickle.dump(marker_flow_list, open(f"marker_flow_list_{time.strftime('%Y_%m_%d_%H_%M_%S')}", "wb"))
    print(f"Success rate: {succeed_num / exp_num}")
    print(f"average_step: {average_step:.2f}")
    env.close()
