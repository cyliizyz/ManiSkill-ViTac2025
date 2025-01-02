from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy

from solutions.actor_and_critics import CustomCritic
from solutions.feature_extractors import FeatureExtractorState


class TD3PolicyForPegInsertionV2(TD3Policy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(TD3PolicyForPegInsertionV2, self).__init__(*args, **kwargs)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None):

        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, FeatureExtractorState(self.observation_space)
        )
        return CustomCritic(**critic_kwargs).to(self.device)
