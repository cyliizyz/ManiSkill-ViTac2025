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
from envs.peg_insertion import PegInsertionSimMarkerFLowEnv
from path import Path
from stable_baselines3.common.utils import set_random_seed
from utils.common import get_time, get_average_params
from loguru import logger as log


EVAL_CFG_FILE = os.path.join(
    track_path, "configs/evaluation/peg_insertion_evaluation.yaml"
)

PEG_NUM = 3
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
            "modified": [],
            "added": [],
            "deleted": [],
            "renamed": [],
        }

        for diff_item in diff_index:
            if diff_item.change_type == "M":
                changed_since_clone["modified"].append(diff_item.a_path)
            elif diff_item.change_type == "A":
                changed_since_clone["added"].append(diff_item.a_path)
            elif diff_item.change_type == "D":
                changed_since_clone["deleted"].append(diff_item.a_path)
            elif diff_item.change_type == "R":
                changed_since_clone["renamed"].append(
                    (diff_item.a_path, diff_item.b_path)
                )

        return {
            "branch": repo.active_branch.name,
            "commit": repo.head.commit.hexsha,
            "commit_message": repo.head.commit.message.strip(),
            "is_dirty": repo.is_dirty(),
            "untracked_files": repo.untracked_files,
            "modified_files": [item.a_path for item in repo.index.diff(None)],
            "changes_since_clone": changed_since_clone,
            "initial_commit": initial_commit.hexsha,
            "initial_commit_message": initial_commit.message.strip(),
        }
    except Exception as e:
        return {"error": f"Failed to get git info: {str(e)}"}


def evaluate_policy(model, key):
    exp_start_time = get_time()
    exp_name = f"peg_insertion_{exp_start_time}"
    log_dir = Path(os.path.join(track_path, f"eval_log/{exp_name}"))
    log_dir.makedirs_p()

    log.remove()
    log.add(
        log_dir / f"{exp_name}.log",
        filter=lambda record: record["extra"]["name"] == "main",
    )

    log.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
        level="INFO",
        filter=lambda record: record["extra"]["name"] == "main",
    )
    eval_log = log.bind(name="main")

    eval_log.info(f"#KEY: {key}")
    eval_log.info(f"this is MD5: {get_self_md5()}")

    git_info = get_git_info()
    eval_log.info("Git Repository Information:")
    for key, value in git_info.items():
        eval_log.info(f"  {key}: {value}")

    with open(EVAL_CFG_FILE, "r") as f:
        cfg = yaml.YAML(typ="safe", pure=True).load(f)

    # get simulation and environment parameters
    sim_params = cfg["env"].pop("params")
    env_name = cfg["env"].pop("env_name")

    params_lb, params_ub = parse_params(env_name, sim_params)
    average_params = get_average_params(params_lb, params_ub)
    eval_log.info(f"\n{average_params}")
    eval_log.info(cfg["env"])

    if "max_action" in cfg["env"].keys():
        cfg["env"]["max_action"] = np.array(cfg["env"]["max_action"])

    specified_env_args = copy.deepcopy(cfg["env"])

    specified_env_args.update(
        {
            "params": average_params,
            "params_upper_bound": average_params,
            "log_path": log_dir,
            "logger": log,
            "env_type": "eval",
        }
    )

    # create evaluation environment
    env = PegInsertionSimMarkerFLowEnv(**specified_env_args)
    set_random_seed(0)

    offset_list = [
        [-4.0, -4.0, -8.0],
        [-4.0, -2.0, 2.0],
        [-4.0, 1.0, -6.0],
        [-4.0, 3.0, 6.0],
        [-3.0, -3.0, -2.0],
        [-3.0, -1.0, 8.0],
        [-3.0, 2.0, 2.0],
        [-2.0, -4.0, -6.0],
        [-2.0, -2.0, 4.0],
        [-2.0, 1.0, -2.0],
        [-2.0, 3.0, 8.0],
        [-1.0, -3.0, 0.0],
        [-1.0, 0.0, 6.0],
        [-1.0, 3.0, 4.0],
        [0.0, -3.0, -4.0],
        [0.0, 0.0, 6.0],
        [0.0, 3.0, 4.0],
        [1.0, -3.0, -4.0],
        [1.0, 0.0, -4.0],
        [1.0, 3.0, 0.0],
        [2.0, -3.0, -8.0],
        [2.0, -1.0, 4.0],
        [2.0, 2.0, -4.0],
        [2.0, 4.0, 6.0],
        [3.0, -2.0, 0.0],
        [3.0, 1.0, -8.0],
        [3.0, 3.0, 2.0],
        [4.0, -3.0, -4.0],
        [4.0, -1.0, 6.0],
        [4.0, 2.0, -2.0],
    ]
    test_num = len(offset_list)
    test_result = []

    for jj in range(REPEAT_NUM):
        for kk in range(test_num):
            for ii in range(PEG_NUM):
                eval_log.opt(colors=True).info(
                    f"<blue>#### Test No. {len(test_result) + 1} ####</blue>"
                )
                obs, _ = env.reset(offset_list[kk], peg_idx=ii)
                initial_offset_of_current_episode = obs["gt_offset"]
                eval_log.info(f"Initial offset: {initial_offset_of_current_episode}")
                d, ep_ret, ep_len = False, 0, 0
                while not d:
                    # Take deterministic actions at test time (noise_scale=0)
                    ep_len += 1
                    for obs_k, obs_v in obs.items():
                        obs[obs_k] = torch.from_numpy(obs_v)
                    action = model(obs)
                    action = action.cpu().detach().numpy().flatten()
                    eval_log.info(f"Step {ep_len} Action: {action}")
                    obs, reward, terminated, truncated, info = env.step(action)
                    d = terminated or truncated
                    if "gt_offset" in obs.keys():
                        eval_log.info(f"Offset: {obs['gt_offset']}")
                    eval_log.info(f"info: {info}")
                    eval_log.info(f"Reward: {reward}")
                    ep_ret += reward
                if info["is_success"]:
                    test_result.append([True, ep_len])
                    eval_log.opt(colors=True).info(f"<green>RESULT: SUCCESS</green>")
                else:
                    test_result.append([False, ep_len])
                    eval_log.opt(colors=True).info(f"<d>RESULT: FAIL</d>")

    env.close()
    success_rate = np.sum(np.array([int(v[0]) for v in test_result])) / (
        test_num * PEG_NUM * REPEAT_NUM
    )
    if success_rate > 0:
        avg_steps = (
            np.mean(np.array([int(v[1]) if v[0] else 0 for v in test_result]))
            / success_rate
        )
        eval_log.info(f"#SUCCESS_RATE: {success_rate*100.0:.2f}%")
        eval_log.info(f"#AVG_STEP: {avg_steps:.2f}")
    else:
        eval_log.info(f"#SUCCESS_RATE: 0")
        eval_log.info(f"#AVG_STEP: NA")


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
