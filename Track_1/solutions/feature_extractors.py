import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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

class CriticFeatureExtractorForLongOpenLock(BaseFeaturesExtractor):
    """critic feature extractor for lock env. the input for critic network is the information of key1 and key2."""
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CriticFeatureExtractorForLongOpenLock, self).__init__(observation_space, features_dim=1)
        self._features_dim = 6

    def forward(self, observations) -> torch.Tensor:
        return torch.cat([observations["key1"], observations["key2"]], dim=-1)

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
