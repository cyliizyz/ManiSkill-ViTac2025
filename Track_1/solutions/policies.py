from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import Actor, TD3Policy

from solutions.actor_and_critics import (
    CustomCritic,
    PointNetActor,
    LongOpenLockPointNetActor,

)
from solutions.feature_extractors import (
    CriticFeatureExtractor,
    CriticFeatureExtractorForLongOpenLock,
    FeatureExtractorForPointFlowEnv,
)


class TD3PolicyForPointFlowEnv(TD3Policy):
    def __init__(
        self,
        *args,
        pointnet_in_dim,
        pointnet_out_dim,
        pointnet_batchnorm,
        pointnet_layernorm,
        zero_init_output,
        use_relative_motion: bool,
        **kwargs,
    ):
        self.pointnet_in_dim = pointnet_in_dim
        self.pointnet_out_dim = pointnet_out_dim
        self.pointnet_layernorm = pointnet_layernorm
        self.pointnet_batchnorm = pointnet_batchnorm
        self.zero_init_output = zero_init_output
        self.use_relative_motion = use_relative_motion
        super(TD3PolicyForPointFlowEnv, self).__init__(*args, **kwargs)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, FeatureExtractorForPointFlowEnv(self.observation_space)
        )

        return PointNetActor(
            pointnet_in_dim=self.pointnet_in_dim,
            pointnet_out_dim=self.pointnet_out_dim,
            batchnorm=self.pointnet_batchnorm,
            layernorm=self.pointnet_layernorm,
            zero_init_output=self.zero_init_output,
            use_relative_motion=self.use_relative_motion,
            **actor_kwargs,
        ).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> CustomCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, CriticFeatureExtractor(self.observation_space)
        )

        return CustomCritic(**critic_kwargs).to(self.device)


class TD3PolicyForLongOpenLockPointFlowEnv(TD3Policy):
    def __init__(
        self,
        *args,
        pointnet_in_dim,
        pointnet_out_dim,
        pointnet_batchnorm,
        pointnet_layernorm,
        zero_init_output,
        use_relative_motion: bool,
        **kwargs,
    ):
        self.pointnet_in_dim = pointnet_in_dim
        self.pointnet_out_dim = pointnet_out_dim
        self.pointnet_layernorm = pointnet_layernorm
        self.pointnet_batchnorm = pointnet_batchnorm
        self.use_relative_motion = use_relative_motion
        self.zero_init_output = zero_init_output
        super(TD3PolicyForLongOpenLockPointFlowEnv, self).__init__(*args, **kwargs)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs,
        )
        return LongOpenLockPointNetActor(
            pointnet_in_dim=self.pointnet_in_dim,
            pointnet_out_dim=self.pointnet_out_dim,
            batchnorm=self.pointnet_batchnorm,
            layernorm=self.pointnet_layernorm,
            zero_init_output=self.zero_init_output,
            use_relative_motion=self.use_relative_motion,
            **actor_kwargs,
        ).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> CustomCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs,
            CriticFeatureExtractorForLongOpenLock(self.observation_space),
        )
        return CustomCritic(**critic_kwargs).to(self.device)
