import argparse
import copy
import glob
import math
import pickle
import sys, os
import logging
import time

from ruamel import yaml

from motion_manager_stage import MotionManagerStage
from solutions.policies import TD3PolicyForPointFlowEnv
from utils.save_load_utils import load_from_zip_file

sys.path.append(os.getcwd())

from utils.utils import get_time, try_make_dir

from continuous_insertion_real import ContinuousInsertionRealPointFlowEnvironment

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--policy_file", type=str, default="")
    parser.add_argument("--repeat_num", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="logs")

    args = parser.parse_args()
    return args


def parse_cfg_file(cfg_file_path):
    with open(cfg_file_path, "r") as f:
        cfg = yaml.safe_load(f)

    ret_dict = {
        "env_name": cfg["env"]["env_name"],
        "policy_name": cfg["policy"]["policy_name"],
    }
    if "max_action" in cfg["env"].keys():
        ret_dict["max_action"] = cfg["env"]["max_action"]
    if "pos_offset_range" in cfg["env"].keys() and "rot_offset_range" in cfg["env"].keys():
        ret_dict["max_error"] = [
            cfg["env"]["pos_offset_range"],
            cfg["env"]["pos_offset_range"],
            cfg["env"]["rot_offset_range"],
        ]
    if "pos_x_offset_range" in cfg["env"].keys():
        ret_dict["pos_x_offset_range"] = cfg["env"]["pos_x_offset_range"]
    if "pos_y_offset_range" in cfg["env"].keys():
        ret_dict["pos_y_offset_range"] = cfg["env"]["pos_y_offset_range"]

    if "z_step_size" in cfg["env"].keys():
        ret_dict["z_step_size"] = cfg["env"]["z_step_size"]
    if "normalize" in cfg["env"].keys():
        ret_dict["normalize"] = cfg["env"]["normalize"]
    if "step_penalty" in cfg["env"].keys():
        ret_dict["penalty"] = cfg["env"]["step_penalty"]
    if "final_reward" in cfg["env"].keys():
        ret_dict["final_reward"] = cfg["env"]["final_reward"]
    if "max_steps" in cfg["env"].keys():
        ret_dict["max_steps"] = cfg["env"]["max_steps"]

    return ret_dict


def evaluate_RL_pointnet_policy(arg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    current_time = get_time()

    cfg = parse_cfg_file(arg.cfg)

    log_subdir = os.path.join(arg.logdir, f"{cfg['env_name']}_{current_time}")
    try_make_dir(log_subdir)
    log_file = logging.FileHandler(os.path.join(log_subdir, f"stdout.log"))
    file_formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    log_file.setFormatter(file_formatter)
    logger.addHandler(log_file)

    if arg.policy_file == "":
        arg.policy_file = os.path.join(os.path.dirname(arg.cfg), "best_model.zip")

    logger.info(f"cfg: {arg.cfg}")
    logger.info(f"weight: {arg.policy_file}")
    logger.info(f"cfg summary: {cfg}")

    env_name, policy_name = cfg.pop("env_name"), cfg.pop("policy_name")
    data, params, _ = load_from_zip_file(arg.policy_file)

    motion_manager = MotionManagerStage("/dev/translation_stage", "/dev/rotation_stage", "/dev/hande")

    env = ContinuousInsertionRealPointFlowEnvironment(motion_manager, clearance=2, **cfg)

    model = TD3PolicyForPointFlowEnv(
        # observation_space=data["observation_space"],
        observation_space=env.observation_space,
        # action_space=data["action_space"],
        action_space=env.action_space,
        lr_schedule=data["lr_schedule"],
        **data["policy_kwargs"],
    )
    model.load_state_dict(params["policy"])
    model.set_training_mode(False)

    ep_ret_total = 0
    test_result = []

    offset_list = [[-4.0, -4.0, -8.0], [-4.0, -2.0, 2.0], [-4.0, 1.0, -6.0], [-4.0, 3.0, 6.0], [-3.0, -3.0, -2.0],
                   [-3.0, -1.0, 8.0], [-3.0, 2.0, 2.0], [-2.0, -4.0, -6.0], [-2.0, -2.0, 4.0], [-2.0, 1.0, -2.0],
                   [-2.0, 3.0, 8.0], [-1.0, -3.0, 0.0], [-1.0, 0.0, 6.0], [-1.0, 3.0, 4.0], [0.0, -3.0, -4.0],
                   [0.0, 0.0, 6.0], [0.0, 3.0, 4.0], [1.0, -3.0, -4.0], [1.0, 0.0, -4.0], [1.0, 3.0, 0.0],
                   [2.0, -3.0, -8.0], [2.0, -1.0, 4.0], [2.0, 2.0, -4.0], [2.0, 4.0, 6.0], [3.0, -2.0, 0.0],
                   [3.0, 1.0, -8.0], [3.0, 3.0, 2.0], [4.0, -3.0, -4.0], [4.0, -1.0, 6.0], [4.0, 2.0, -2.0]]
    logger.info(offset_list)
    offset_list = offset_list * arg.repeat_num
    np.set_printoptions(precision=3, suppress=True)
    failure_cases = []
    all_test_log = []

    for k in range(len(offset_list)):
        logger.info(f"Test No. {k + 1}")
        cur_test_log = []
        o = env.reset(offset_list[k])
        initial_offset_of_current_episode = o["gt_offset"]
        logger.info(f"Initial offset: {initial_offset_of_current_episode}")
        cur_test_log.append(("gt_offset", initial_offset_of_current_episode))
        d, ep_ret, ep_len = False, 0, 0
        while not d:
            # Take deterministic actions at test time (noise_scale=0)
            for obs_k, obs_v in o.items():
                o[obs_k] = torch.from_numpy(obs_v)
            action = model(o)
            action = action.cpu().detach().numpy().flatten()
            # action[1] = - action[1]
            logger.info(f"Action: {action}")
            cur_test_log.append(("action", action))
            o, r, d, info = env.step(action)
            cur_test_log.append(("gt_offset", o["gt_offset"]))
            logger.info(info["message"])
            ep_ret += r
            ep_len += 1
        if info["is_success"]:
            test_result.append([True, ep_len])
            cur_test_log.append(("result", "success"))
        else:
            test_result.append([False, ep_len])
            cur_test_log.append(("result", "fail"))
            failure_cases.append(initial_offset_of_current_episode)
        ep_ret_total += ep_ret

        logger.info("Test total reward %.2f, episode length %d" % (ep_ret, ep_len))
        cur_test_log.append(("total_reward", ep_ret))
        cur_test_log.append(("episode_length", ep_len))
        current_success_rate = np.sum(np.array([int(v[0]) for v in test_result])) / (k + 1)
        if current_success_rate > 0:
            current_mean_ep_len = (
                np.mean(np.array([int(v[1]) if v[0] else 0 for v in test_result])) / current_success_rate
            )
        else:
            current_mean_ep_len = 0
        logger.info(f"current success rate: {current_success_rate:.2f}; current mean ep_len: {current_mean_ep_len}")
        all_test_log.append(cur_test_log)

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in test_result])) / len(offset_list)
    if success_rate > 0:
        avg_steps = np.mean(np.array([int(v[1]) if v[0] else 0 for v in test_result])) / success_rate
    else:
        avg_steps = 0

    result_string = (
        f"-----------TEST--SUMMARY-----------\n"
        f"cfg: {arg.cfg}\n"
        f"weight: {arg.policy_file}\n"
        f"average_test_reward: {ep_ret_total / len(offset_list)}\n"
        f"Success Rate: {success_rate}, Average step num: {avg_steps}"
    )
    logger.info(result_string)

    all_test_log_path = os.path.join(log_subdir, "all_test_log.pkl")
    with open(all_test_log_path, "wb") as f:
        pickle.dump(all_test_log, f)
    logger.info(f"All test log saved in {all_test_log_path}.")
    if len(failure_cases) > 0:
        failure_cases_path = os.path.join(log_subdir, "failure_cases.npy")
        np.save(failure_cases_path, np.array(failure_cases))

        logger.info(f"Failure cases saved in {failure_cases_path}.")

    logger.removeHandler(stream_handler)
    logger.removeHandler(log_file)
    return success_rate, avg_steps


if __name__ == "__main__":
    args = parse_args()

    cfg_files = ["/home/user/trained_policy_with_gelsight_mini/cfg.yaml"]

    print("There are", len(cfg_files), "cfg files to test: ")
    print(cfg_files)

    for cfg_file in cfg_files:
        args.cfg = cfg_file
        evaluate_RL_pointnet_policy(copy.deepcopy(args))

