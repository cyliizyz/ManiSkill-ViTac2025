from typing import List, Tuple, Type

import gymnasium as gym
import torch
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.td3.policies import Actor
from torch import nn

from solutions.networks import PointNetFeatureExtractor


class PointNetActor(Actor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        features_extractor: nn.Module,
        pointnet_in_dim: int,
        pointnet_out_dim: int,
        normalize_images: bool = True,
        batchnorm=False,
        layernorm=True,
        zero_init_output=False,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            **kwargs,
        )

        action_dim = get_action_dim(self.action_space)

        self.point_net_feature_extractor = PointNetFeatureExtractor(
            dim=pointnet_in_dim, out_dim=pointnet_out_dim, batchnorm=batchnorm
        )

        self.mlp_policy = nn.Sequential(
            nn.Linear(pointnet_out_dim * 2, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        if zero_init_output:
            last_linear = None
            for m in self.mlp_policy.children():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.zeros_(last_linear.bias)
                last_linear.weight.data.copy_(0.01 * last_linear.weight.data)

        # self.mu = nn.Sequential(*actor_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            marker_pos = self.extract_features(obs, self.features_extractor)

        if marker_pos.ndim == 3:
            marker_pos = torch.unsqueeze(marker_pos, dim=0)

        batch_num = marker_pos.shape[0]
        l_marker_pos = marker_pos[:, 0, ...]
        r_marker_pos = marker_pos[:, 1, ...]
        marker_pos_input = torch.cat([l_marker_pos, r_marker_pos], dim=0)
        point_flow_fea = self.point_net_feature_extractor(marker_pos_input)
        l_point_flow_fea = point_flow_fea[:batch_num, ...]
        r_point_flow_fea = point_flow_fea[batch_num:, ...]
        point_flow_fea = torch.cat([l_point_flow_fea, r_point_flow_fea], dim=-1)
        pred = self.mlp_policy(point_flow_fea)

        return pred


class CustomCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            **kwargs,
        )

        action_dim = get_action_dim(self.action_space)
        self.features_dim = features_dim

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = nn.Sequential(
                *create_mlp(self.features_dim + action_dim, 1, net_arch, activation_fn)
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(False):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](torch.cat([features, actions], dim=1))
