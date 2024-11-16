import copy
import hashlib
import os
import sys
import numpy as np
import ruamel.yaml as yaml
import torch
from stable_baselines3.common.save_util import load_from_zip_file
import git

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

from scripts.arguments import parse_params
from envs.peg_insertion_v2 import PegInsertionSimMarkerFLowEnvV2
from path import Path
from stable_baselines3.common.utils import set_random_seed
from utils.common import get_time, get_average_params
from loguru import logger

from scripts.arguments import parse_params, handle_policy_args


EVAL_CFG_FILE = os.path.join(
    track_path, "configs/evaluation/peg_insertion_v2_evaluation.yaml"
)
REPEAT_NUM = 3


def get_self_md5():
    script_path = os.path.abspath(sys.argv[0])

    md5_hash = hashlib.md5()
    with open(script_path, "rb") as f:

        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def get_git_info():
    """Get git repository information including changes since clone."""
    try:
        repo = git.Repo(search_parent_directories=True)
        # Get the initial commit (first commit after clone)
        initial_commit = list(repo.iter_commits())[-1]
        
        # Get diff between initial commit and current state
        diff_index = initial_commit.diff(None)
        
        changed_since_clone = {
            'modified': [],
            'added': [],
            'deleted': [],
            'renamed': []
        }
        
        for diff_item in diff_index:
            if diff_item.change_type == 'M':
                changed_since_clone['modified'].append(diff_item.a_path)
            elif diff_item.change_type == 'A':
                changed_since_clone['added'].append(diff_item.a_path)
            elif diff_item.change_type == 'D':
                changed_since_clone['deleted'].append(diff_item.a_path)
            elif diff_item.change_type == 'R':
                changed_since_clone['renamed'].append((diff_item.a_path, diff_item.b_path))

        return {
            'branch': repo.active_branch.name,
            'commit': repo.head.commit.hexsha,
            'commit_message': repo.head.commit.message.strip(),
            'is_dirty': repo.is_dirty(),
            'untracked_files': repo.untracked_files,
            'modified_files': [item.a_path for item in repo.index.diff(None)],
            'changes_since_clone': changed_since_clone,
            'initial_commit': initial_commit.hexsha,
            'initial_commit_message': initial_commit.message.strip()
        }
    except Exception as e:
        return {
            'error': f"Failed to get git info: {str(e)}"
        }


def evaluate_policy(model, key):
    exp_start_time = get_time()
    exp_name = f"peg_insertion_v2_{exp_start_time}"
    log_dir = Path(os.path.join(track_path, f"eval_log/{exp_name}"))
    log_dir.makedirs_p()

    logger.remove()
    logger.add(log_dir / f"{exp_name}.log")
    logger.add(
        sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}", level="INFO"
    )

    logger.info(f"#KEY: {key}")
    logger.info(f"this is MD5: {get_self_md5()}")
    
    # Add git information logging
    git_info = get_git_info()
    logger.info("Git Repository Information:")
    for key, value in git_info.items():
        logger.info(f"  {key}: {value}")

    with open(EVAL_CFG_FILE, "r") as f:
        cfg = yaml.YAML(typ="safe", pure=True).load(f)

    # get simulation and environment parameters
    sim_params = cfg["env"].pop("params")
    env_name = cfg["env"].pop("env_name")

    params_lb, params_ub = parse_params(env_name, sim_params)
    average_params = get_average_params(params_lb, params_ub)
    logger.info(f"\n{average_params}")
    logger.info(cfg["env"])

    if "max_action" in cfg["env"].keys():
        cfg["env"]["max_action"] = np.array(cfg["env"]["max_action"])

    specified_env_args = copy.deepcopy(cfg["env"])

    specified_env_args.update(
        {
            "params": average_params,
            "params_upper_bound": average_params,
        }
    )

    # create evaluation environment
    env = PegInsertionSimMarkerFLowEnvV2(**specified_env_args)
    set_random_seed(0)

    offset_list = [
        [2.3, -4.1, -3.2, 8.8, 1],
        [-3.0, 0.5, -6.3, 7.9, 1],
        [3.2, 0.5, 4.2, 3.4, 1],
        [-1.3, 2.1, 9.1, 7.9, 0],
        [-1.9, 0.3, -1.9, 5.4, 1],
        [-1.5, 1.0, -9.0, 4.9, 1],
        [1.8, -2.4, 5.0, 8.9, 1],
        [1.1, -2.7, 1.9, 3.6, 0],
        [-4.8, -4.0, 5.8, 6.3, 0],
        [-3.2, 1.1, 9.0, 8.8, 1],
        [1.2, 0.2, -3.2, 6.8, 0],
        [4.2, -4.9, -5.6, 4.8, 1],
        [4.8, 3.6, 4.9, 8.4, 0],
        [-4.2, -2.7, -3.3, 5.8, 0],
        [-4.0, 2.2, 3.7, 5.3, 1],
        [1.6, 0.3, -6.5, 5.7, 1],
        [3.6, -3.7, 0.2, 6.7, 1],
        [-3.0, 3.8, -3.8, 4.2, 0],
        [1.9, -2.4, 1.7, 3.1, 0],
        [1.4, -2.5, -6.1, 5.4, 1],
        [1.7, 1.5, 9.1, 8.2, 1],
        [-1.2, 0.1, 8.8, 8.6, 0],
        [-3.5, 3.5, 1.4, 4.5, 1],
        [4.9, -4.2, -3.9, 5.7, 1],
        [-4.2, 4.7, -4.8, 5.7, 0],
        [3.2, -2.3, 7.9, 5.8, 0],
        [-2.8, 0.8, 0.1, 7.2, 1],
        [0.8, -3.6, 0.5, 3.9, 0],
        [-3.6, 0.0, -0.9, 5.3, 0],
        [3.5, 4.7, -7.3, 3.4, 1],
        [-3.4, -1.4, 1.5, 4.5, 0],
        [-0.7, 4.6, 7.7, 7.9, 1],
        [4.9, -1.6, -3.2, 8.6, 1],
        [3.2, -0.1, 1.2, 5.1, 0],
        [3.3, 4.6, -5.5, 3.6, 1],
        [2.9, 4.1, 3.8, 4.2, 0],
        [0.1, 5.0, -9.2, 3.2, 0],
        [4.1, -3.9, 6.3, 7.9, 1],
        [-1.2, -3.8, 7.8, 7.8, 1],
        [0.7, -4.0, -3.2, 7.0, 0],
        [-1.2, -4.5, -2.7, 6.3, 1],
        [-4.5, 2.1, -2.5, 5.4, 1],
        [0.0, 2.3, 9.1, 4.0, 0],
        [0.2, 1.7, -6.4, 4.5, 1],
        [-2.7, -3.4, 7.5, 4.9, 1],
        [3.6, 0.3, 8.8, 8.6, 1],
        [-2.9, -0.6, 2.6, 6.5, 0],
        [-3.4, -1.9, 4.1, 6.6, 1],
        [2.3, 2.9, -8.5, 6.6, 1],
        [-3.9, 1.8, 9.9, 8.4, 1],
    ]

    test_num = len(offset_list)
    test_result = []
    for ii in range(test_num):
        for kk in range(REPEAT_NUM):
            logger.opt(colors=True).info(
                f"<blue>#### Test No. {len(test_result) + 1} ####</blue>"
            )
            o, _ = env.reset(offset_list[ii])
            initial_offset_of_current_episode = o["gt_offset"]
            logger.info(f"Initial offset: {initial_offset_of_current_episode}")
            d, ep_ret, ep_len = False, 0, 0
            while not d:
                # Take deterministic actions at test time (noise_scale=0)
                ep_len += 1
                for obs_k, obs_v in o.items():
                    o[obs_k] = torch.from_numpy(obs_v)
                action = model(o)
                action = action.cpu().detach().numpy().flatten()
                logger.info(f"Step {ep_len} Action: {action}")
                o, r, terminated, truncated, info = env.step(action)
                d = terminated or truncated
                if "gt_offset" in o.keys():
                    logger.info(f"Offset: {o['gt_offset']}")
                if "surface_diff" in o.keys():
                    logger.info(f"Surface Diff: {o['surface_diff']}")
                logger.info(f"info: {info}")
                logger.info(f"Reward: {r}")
                ep_ret += r
            if info["is_success"]:
                test_result.append([True, ep_len])
                logger.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
            else:
                test_result.append([False, ep_len])
                logger.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in test_result])) / (
        test_num * REPEAT_NUM
    )
    if success_rate > 0:
        avg_steps = (
            np.mean(np.array([int(v[1]) if v[0] else 0 for v in test_result]))
            / success_rate
        )
        logger.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        logger.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        logger.info(f"#SUCCESS_RATE: 0")
        logger.info(f"#AVG_STEP: NA")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--team_name", type=str, required=True, help="your team name")
    parser.add_argument("--model_name", required=True, help="your model")
    parser.add_argument("--policy_file_path", required=True, help="your best_model")

    args = parser.parse_args()
    team_name = args.team_name
    model_name = args.model_name
    policy_file = args.policy_file_path

    data, params, _ = load_from_zip_file(policy_file)

    from solutions import policies

    model_class = getattr(policies, model_name)
    model = model_class(
        observation_space=data["observation_space"],
        action_space=data["action_space"],
        lr_schedule=data["lr_schedule"],
        **data["policy_kwargs"],
    )
    model.load_state_dict(params["policy"])
    evaluate_policy(model, team_name)
