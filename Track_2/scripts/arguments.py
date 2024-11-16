import argparse
import copy
import os
import sys

import numpy as np
from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

from envs.common_params import CommonParams
from envs.peg_insertion_v2 import PegInsertionParams

from solutions.feature_extractors import *


def get_parser():
    parser = argparse.ArgumentParser(description="universal training script")

    parser.add_argument(
        "--cfg",
        type=str,
        help="specify the config file for the run",
    )

    # parameters should be defined in the yaml file
    # but can be overwritten by command line arguments

    #### RL parameters ####
    parser.add_argument("--checkpoint_every", type=int)
    parser.add_argument("--buffer_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--learning_starts", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_freq", type=int)
    parser.add_argument("--policy_delay", type=int)
    parser.add_argument("--gradient_steps", type=int)
    parser.add_argument("--total_timesteps", type=int)
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--eval_freq", type=int)
    parser.add_argument("--n_eval", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_render", action="store_true", help="renderless mode")
    return parser


def solve_argument_conflict(cmd_arg, dict_arg):
    policy_args = [
        "policy_name",
        "policy_kwargs",
        "buffer_size",
        "train_freq",
        "gradient_steps",
        "learning_starts",
        "target_policy_noise",
        "target_noise_clip",
        "action_noise",
        "batch_size",
        "learning_rate",
        "policy_delay",
    ]
    train_args = [
        "name",
        "total_timesteps",
        "log_interval",
        "checkpoint_every",
        "eval_freq",
        "n_eval",
        "parallel",
        "timeout",
        "affinity_num_each_process",
        "affinity_offset",
        "seed",
        "gpu",
        "project_name",
    ]

    for key, value in cmd_arg.__dict__.items():
        if value is None:
            continue
        if key in policy_args:
            dict_arg["policy"][key] = cmd_arg.__dict__[key]
            print(f"policy arg: {key} is set to {value}")
        elif key in train_args:
            if key == "name":
                dict_arg["train"]["name"] = f"{dict_arg['train']['name']}_{value}"
            else:
                dict_arg["train"][key] = value
            print(f"train arg: {key} is set to {value}")
        else:
            dict_arg[key] = cmd_arg.__dict__[key]
            print(f"other arg: {key} is set to {value}")

    return dict_arg


def parse_params(environment_name, params):
    if "Peg" in environment_name:
        params_lowerbound: CommonParams = PegInsertionParams()

    else:
        raise NotImplementedError

    params_upperbound = copy.deepcopy(params_lowerbound)
    for key, value in params.items():
        if key in params_lowerbound.__dict__:
            if type(value) is not list:
                value = [value, value]
            params_lowerbound.__dict__[key] = value[0]
            params_upperbound.__dict__[key] = value[-1]

    return params_lowerbound, params_upperbound


feature_extractor_classes = {
    "State": FeatureExtractorState,
    "PointCloud": FeaturesExtractorPointCloud,
}


def handle_policy_args(original_cfg, log_dir, action_dim=3):
    original_cfg["policy"]["device"] = original_cfg["train"]["device"]

    original_cfg["policy"]["action_noise"] = VectorizedActionNoise(
        NormalActionNoise(
            np.array([0] * action_dim),
            np.array([original_cfg["policy"]["action_noise"]] * action_dim),
        ),
        original_cfg["train"]["parallel"],
    )

    original_cfg["policy"]["seed"] = original_cfg["train"]["seed"]

    original_cfg["policy"]["tensorboard_log"] = log_dir

    if "policy_kwargs" in original_cfg["policy"].keys():

        if "encoder_weight" in original_cfg["policy"]["policy_kwargs"]:
            if not original_cfg["policy"]["policy_kwargs"]["encoder_weight"].startswith(
                "/"
            ):
                original_cfg["policy"]["policy_kwargs"]["encoder_weight"] = (
                    os.path.join(
                        track_path,
                        original_cfg["policy"]["policy_kwargs"]["encoder_weight"],
                    )
                )

        if "decoder_weight" in original_cfg["policy"]["policy_kwargs"]:
            if not original_cfg["policy"]["policy_kwargs"]["decoder_weight"].startswith(
                "/"
            ):
                original_cfg["policy"]["policy_kwargs"]["decoder_weight"] = (
                    os.path.join(
                        track_path,
                        original_cfg["policy"]["policy_kwargs"]["decoder_weight"],
                    )
                )

    # new
    features_extractor_class_name = original_cfg["policy"]["policy_kwargs"].pop(
        "features_extractor_class"
    )
    if features_extractor_class_name is not None:
        original_cfg["policy"]["policy_kwargs"]["features_extractor_class"] = (
            feature_extractor_classes[features_extractor_class_name]
        )

    return original_cfg
