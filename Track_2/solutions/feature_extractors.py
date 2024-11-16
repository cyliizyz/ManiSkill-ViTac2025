import os, sys
script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from networks import PointNetFeatureExtractor

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import gymnasium as gym
from solutions.networks import PointNetFeatureExtractor

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision.transforms as transforms

"""
Feature Extractors for different environments
by default, the feature extractors are for actor network
unless it starts with "CriticFeatureExtractor"
"""


class CriticFeatureExtractor(BaseFeaturesExtractor):
    """general critic feature extractor for peg-in-hole env. the input for critic network is the gt_offset."""

    def __init__(self, observation_space: gym.spaces):
        super(CriticFeatureExtractor, self).__init__(observation_space, features_dim=3)
        self._features_dim = 3

    def forward(self, observations) -> torch.Tensor:
        return observations["gt_offset"]

class FeatureExtractorForPointFlowEnv(BaseFeaturesExtractor):
    """
    feature extractor for point flow env. the input for actor network is the point flow.
    so this 'feature extractor' actually only extracts point flow from the original observation dictionary.
    the actor network contains a pointnet module to extract latent features from point flow.
    """

    def __init__(self, observation_space: gym.spaces):
        super(FeatureExtractorForPointFlowEnv, self).__init__(observation_space, features_dim=512)
        self._features_dim = 512

    def forward(self, observations) -> torch.Tensor:
        original_obs = observations["marker_flow"]
        if original_obs.ndim == 4:
            original_obs = torch.unsqueeze(original_obs, 0)
        # (batch_num, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v))
        fea = torch.cat([original_obs[:, :, 0, ...], original_obs[:, :, 1, ...]], dim=-1)
        return fea

class FeatureExtractorState(BaseFeaturesExtractor):
    """
    General critic feature extractor for PegInsertion env (v2).
    The input for critic network is the gt_offset + relative_motion + direction.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        super(FeatureExtractorState, self).__init__(observation_space, features_dim=9)

    def forward(self, observations) -> torch.Tensor:
        gt_offset = observations["gt_offset"]  # 4
        relative_motion = observations["relative_motion"]  # 4 
        gt_direction = observations["gt_direction"] # 1
        return torch.cat([gt_offset, relative_motion, gt_direction], dim=-1)

class FeaturesExtractorPointCloud(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,
                 vision_kwargs:dict=None,
                 tac_kwargs:dict=None):
        super().__init__(observation_space, features_dim=1)

        # PointCloud
        vision_dim = vision_kwargs.get('dim', 3)
        vision_out_dim = vision_kwargs.get('out_dim', 64)
        self.vision_scale = vision_kwargs.get('scale', 1.0)
        vision_batchnorm = vision_kwargs.get('batchnorm', False)
        self.point_net_vision1 = PointNetFeatureExtractor(dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm)
        self.point_net_vision2 = PointNetFeatureExtractor(dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm)
        self.vision_feature_dim = vision_out_dim * 2
        # Tactile
        tac_dim = tac_kwargs.get('dim', 4)
        tac_out_dim = tac_kwargs.get('out_dim', 32)
        tac_batchnorm = tac_kwargs.get('batchnorm', False)
        self.point_net_tac = PointNetFeatureExtractor(dim=tac_dim, out_dim=tac_out_dim, batchnorm=tac_batchnorm)
        self.tac_feature_dim = tac_out_dim * 2

        self.layernorm_vision = nn.LayerNorm(self.vision_feature_dim)
        self.layernorm_tac = nn.LayerNorm(self.tac_feature_dim)

        self._features_dim = self.vision_feature_dim + self.tac_feature_dim

    def parse_obs(self, obs:dict):
        obs = obs.copy()
        marker_flow = obs["marker_flow"]
        point_cloud = torch.Tensor(obs["object_point_cloud"])

        unsqueezed = False
        if marker_flow.ndim == 4:
            assert point_cloud.ndim == 3
            marker_flow = torch.unsqueeze(marker_flow, 0)
            point_cloud = point_cloud.unsqueeze(0)
            unsqueezed = True

        fea = torch.cat([marker_flow[:, :, 0, ...], marker_flow[:, :, 1, ...]], dim=-1)
        tactile_left, tactile_right = fea[:, 0], fea[:, 1]  # (batch_size, marker_num, 4[u0,v0,u1,v1])

        point_cloud = point_cloud * self.vision_scale

        return tactile_left, tactile_right, point_cloud,  unsqueezed

    def forward(self, obs):
        tactile_left, tactile_right, point_cloud, unsqueezed = self.parse_obs(obs)

        # the gripper is ignored here.
        vision_feature_1 = self.point_net_vision1(point_cloud[:, 0])  # object 1
        vision_feature_2 = self.point_net_vision2(point_cloud[:, 1])  # object 2
        vision_feature = torch.cat([vision_feature_1, vision_feature_2], dim=-1)

        tactile_left_feature = self.point_net_tac(tactile_left)
        tactile_right_feature = self.point_net_tac(tactile_right)
        tactile_feature = torch.cat([tactile_left_feature, tactile_right_feature], dim=-1)

        features = torch.cat([self.layernorm_vision(vision_feature), self.layernorm_tac(tactile_feature)], dim=-1)
        if unsqueezed:
            features = features.squeeze(0)
        return features

