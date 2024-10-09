import itertools
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (get_parameters_by_name,
                                            polyak_update)
from stable_baselines3.td3.policies import TD3Policy
from torch import nn
from torch.nn import functional as F

from Track_1.solutions.feature_extractors import CriticFeatureExtractor, CriticFeatureExtractorForLongOpenLock
from Track_1.solutions.networks import PointNetFeatureExtractor

"""
modules in this file are different from the others in the same directory.
this file is written to use the pretrained pointnet latent feature encoder in policy training.
so this feature extractor can extract the latent feature from the marker flow, which is different from those in feature_extractors.py.
"""

class FeatureExtractorWithPointNetEncoder(BaseFeaturesExtractor):
    """
    this feature extractor can extract the latent feature from the marker flow, which is different from those in feature_extractors.py.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=64):
        super(FeatureExtractorWithPointNetEncoder, self).__init__(observation_space, features_dim=features_dim)
        self._features_dim = features_dim
        pointnet_in_dim = observation_space["marker_flow"].shape[-1] * 2
        self.feature_extractor_net = PointNetFeatureExtractor(dim=pointnet_in_dim, out_dim=int(features_dim / 2))

    def forward(self, observations) -> torch.Tensor:
        original_obs = observations["marker_flow"]
        if original_obs.ndim == 4:
            original_obs = torch.unsqueeze(original_obs, 0)
        batch_num = original_obs.shape[0]
        # (batch_num, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v))
        feature_extractor_input = torch.cat([original_obs[:, :, 0, ...], original_obs[:, :, 1, ...]], dim=-1)
        feature_extractor_input = torch.cat([feature_extractor_input[:, 0, ...], feature_extractor_input[:, 1, ...]], dim=0)
        # (batch_num * 2, 128, 4)
        # l_marker_pos = feature_extractor_input[:, 0, ...]
        # r_marker_pos = feature_extractor_input[:, 1, ...]
        # shape: (batch, num_points, 4)

        # with torch.inference_mode():
        # self.point_net_feature_extractor.eval()
        marker_flow_fea = self.feature_extractor_net(feature_extractor_input)
        # l_marker_flow_fea = self.feature_extractor_net(l_marker_pos)
        # r_marker_flow_fea = self.feature_extractor_net(r_marker_pos)  # (batch_num, pointnet_feature_dim)
        marker_flow_fea = torch.cat([marker_flow_fea[:batch_num], marker_flow_fea[batch_num:]], dim=-1)

        return marker_flow_fea

class PointNetDecoder(nn.Module):
    def __init__(self, latent_dim, point_dim, point_num):
        super(PointNetDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.point_dim = point_dim
        self.point_num = point_num

        self.conv_decoder = nn.Sequential(
            nn.Conv1d(latent_dim + point_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, point_dim, 1),
        )

    def forward(self, initial_pos, x):
        """
        :param initial_pos: initial marker positions of shape (batchsize, point_num, point_dim)
        :param x: latent vector of shape (latent_dim,)
        """
        initial_pos = torch.permute(initial_pos, (0, 2, 1))
        latent = x.unsqueeze(-1).repeat(1, 1, self.point_num)
        combined = torch.cat([initial_pos, latent], dim=1)
        reconstructed = self.conv_decoder(combined)
        reconstructed = torch.permute(reconstructed, (0, 2, 1))
        return reconstructed

class CustomActor(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        layernorm=True,
        zero_init_output=True,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            **kwargs,
        )

        action_dim = get_action_dim(self.action_space)

        self.mlp_policy = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.zero_init_output = zero_init_output
        if zero_init_output:
            last_linear = None
            for m in self.mlp_policy.children():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.zeros_(last_linear.bias)
                last_linear.weight.data.copy_(0.01 * last_linear.weight.data)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        point_flow_fea = self.extract_features(obs)
        pred = self.mlp_policy(point_flow_fea)
        return pred

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)

class TD3PolicyWithPointNetEncoder(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        pointnet_out_dim=32,
        layernorm=True,
        zero_init_output=True,
        encoder_weight=None,
        decoder_weight=None,
        # normalize_features=False,
        activation_fn: Type[nn.Module] = nn.ReLU,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=None,
            features_extractor_kwargs=None,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        critic_arch = net_arch["qf"]
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update(
            {
                "zero_init_output": zero_init_output,
                "layernorm": layernorm,
                # "normalize_features": normalize_features,
            }
        )
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "normalize_images": False,
                "n_critics": 2,
                "net_arch": critic_arch,
                "activation_fn": activation_fn,
                "share_features_extractor": False,
            }
        )
        self.activation_fn = activation_fn
        self.pointnet_out_dim = pointnet_out_dim
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight

        self._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.point_dim = self.observation_space["marker_flow"].shape[-1]
        self.point_num = self.observation_space["marker_flow"].shape[-2]
        self.actor = self.make_actor(
            features_extractor=FeatureExtractorWithPointNetEncoder(
                self.observation_space, features_dim=self.pointnet_out_dim * 2
            )
        )
        self.actor_target = self.make_actor(
            features_extractor=FeatureExtractorWithPointNetEncoder(
                self.observation_space, features_dim=self.pointnet_out_dim * 2
            )
        )

        if self.encoder_weight and os.path.exists(self.encoder_weight):
            with open(self.encoder_weight, "rb") as f:
                self.actor.features_extractor.feature_extractor_net.load_state_dict(torch.load(f))

        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        # actor optimizer does not update the features extractor
        self.actor.optimizer = self.optimizer_class(
            self.actor.mlp_policy.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        # Create new features extractor for each network
        self.critic = self.make_critic(features_extractor=CriticFeatureExtractor(self.observation_space))
        self.critic_target = self.make_critic(features_extractor=CriticFeatureExtractor(self.observation_space))

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

        self.pointnet_decoder = PointNetDecoder(
            latent_dim=self.pointnet_out_dim, point_dim=self.point_dim, point_num=self.point_num
        )
        if self.decoder_weight and os.path.exists(self.decoder_weight):
            with open(self.decoder_weight, "rb") as f:
                self.pointnet_decoder.load_state_dict(torch.load(f))

        self.features_extractor_optimizer = self.optimizer_class(
            itertools.chain(self.actor.features_extractor.parameters(), self.pointnet_decoder.parameters()),
            lr=lr_schedule(1),
            weight_decay=1e-5,
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
            )
        )
        return data

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

class TD3PolicyWithPointNetEncoderForLongOpenLock(TD3PolicyWithPointNetEncoder):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        pointnet_out_dim=32,
        layernorm=True,
        zero_init_output=True,
        encoder_weight=None,
        decoder_weight=None,
        # normalize_features=False,
        activation_fn: Type[nn.Module] = nn.ReLU,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            pointnet_out_dim=pointnet_out_dim,
            layernorm=layernorm,
            zero_init_output=zero_init_output,
            encoder_weight=encoder_weight,
            decoder_weight=decoder_weight,
            # normalize_features=normalize_features,
            activation_fn=activation_fn,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.point_dim = self.observation_space["marker_flow"].shape[-1]
        self.point_num = self.observation_space["marker_flow"].shape[-2]
        self.actor = self.make_actor(
            features_extractor=FeatureExtractorWithPointNetEncoder(
                self.observation_space, features_dim=self.pointnet_out_dim * 2
            )
        )
        self.actor_target = self.make_actor(
            features_extractor=FeatureExtractorWithPointNetEncoder(
                self.observation_space, features_dim=self.pointnet_out_dim * 2
            )
        )

        if self.encoder_weight and os.path.exists(self.encoder_weight):
            with open(self.encoder_weight, "rb") as f:
                self.actor.features_extractor.feature_extractor_net.load_state_dict(torch.load(f))

        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        # actor optimizer does not update the features extractor
        self.actor.optimizer = self.optimizer_class(
            self.actor.mlp_policy.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        # Create new features extractor for each network
        self.critic = self.make_critic(features_extractor=CriticFeatureExtractorForLongOpenLock(self.observation_space))
        self.critic_target = self.make_critic(features_extractor=CriticFeatureExtractorForLongOpenLock(self.observation_space))

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

        self.pointnet_decoder = PointNetDecoder(
            latent_dim=self.pointnet_out_dim, point_dim=self.point_dim, point_num=self.point_num
        )
        if self.decoder_weight and os.path.exists(self.decoder_weight):
            with open(self.decoder_weight, "rb") as f:
                self.pointnet_decoder.load_state_dict(torch.load(f))

        self.features_extractor_optimizer = self.optimizer_class(
            itertools.chain(self.actor.features_extractor.parameters(), self.pointnet_decoder.parameters()),
            lr=lr_schedule(1),
            weight_decay=1e-5,
            **self.optimizer_kwargs,
        )


CustomTD3Self = TypeVar("CustomTD3Self", bound="CustomTD3")

class CustomTD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically (Only available when passing string for the environment).
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        reconstruction_learning_rate: float = 5e-5,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        reconstruction_train_freq_multiplier: int = 1,
        reconstruction_train_loss_threshold: float = 0,
        reconstruction_train_max_steps: int = -1,
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        update_actor_after: int = 0,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )
        self.update_actor_after = update_actor_after
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

        self.reconstruction_loss_fn = nn.MSELoss()
        self.reconstruction_learning_rate = reconstruction_learning_rate
        self.reconstruction_train_freq_multiplier = reconstruction_train_freq_multiplier
        self.reconstruction_train_loss_threshold = reconstruction_train_loss_threshold
        self.reconstruction_train_max_steps = reconstruction_train_max_steps
        self.train_reconstruction = True if self.reconstruction_train_freq_multiplier > 0 else False

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        for param_group in self.policy.features_extractor_optimizer.param_groups:
            param_group["lr"] = self.reconstruction_learning_rate

        actor_losses, critic_losses = [], []
        reconstruction_losses = []
        time_critic, time_sample, time_1s, time_2s, time_3s, time_4s = [], [], [], [], [], []
        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            time_sample_start = time.time()
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            time_0 = time.time()
            time_sample.append((time_0 - time_sample_start) * 1000)
            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            time_critic.append((time.time() - time_0) * 1000)

            if self.train_reconstruction:
                # Update feature extractor and decoder
                # self.actor.disable_mlp_grad()
                for __ in range(self.reconstruction_train_freq_multiplier):
                    reconstruction_replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                    reconstruction_loss = self.calculate_reconstruction_loss(reconstruction_replay_data.observations)
                    reconstruction_losses.append(reconstruction_loss.item())
                    self.policy.features_extractor_optimizer.zero_grad()
                    reconstruction_loss.backward()
                    self.policy.features_extractor_optimizer.step()
                polyak_update(self.actor.features_extractor.parameters(), self.actor_target.features_extractor.parameters(), 1)
                # update encoder parameters here to save some time
                # also it is fully updated
                if (self.reconstruction_train_max_steps > 0 and self.reconstruction_train_loss_threshold > 0) and (
                    self._n_updates >= self.reconstruction_train_max_steps
                    or reconstruction_losses[-1] < self.reconstruction_train_loss_threshold
                ):
                    self.train_reconstruction = (
                        False  # Stop training reconstruction after threshold or max steps is reached
                    )

            # Delayed policy updates
            if self._n_updates > self.update_actor_after and self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                # self.actor.disable_feature_extractor_grad()
                time_1 = time.time()
                with torch.no_grad():
                    features = self.actor.features_extractor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor.mlp_policy(features)
                ).mean()
                actor_losses.append(actor_loss.item())
                time_2 = time.time()
                time_1s.append((time_2 - time_1) * 1000)
                # Optimize the actor
                # weight_before_optimization = copy.deepcopy(self.actor.mlp_policy.state_dict())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                time_3 = time.time()
                time_2s.append((time_3 - time_2) * 1000)
                self.actor.optimizer.step()
                time_4 = time.time()
                time_3s.append((time_4 - time_3) * 1000)
                # weight_after_optimization = copy.deepcopy(self.actor.mlp_policy.state_dict())
                # weight_difference = 0
                # for key, value in weight_before_optimization.items():
                #     weight_difference += torch.sum(torch.abs(value - weight_after_optimization[key]))
                # print(weight_difference)

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.mlp_policy.parameters(), self.actor_target.mlp_policy.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
                time_4s.append((time.time() - time_4) * 1000)

        print("sample: ", len(time_sample), np.mean(time_sample), np.sum(time_sample))
        print("critic: ", len(time_critic), np.mean(time_critic), np.sum(time_critic))
        print("forward: ", len(time_1s),  np.mean(time_1s), np.sum(time_1s))
        print("backward: ", len(time_2s), np.mean(time_2s), np.sum(time_2s))
        print("optimizer step: ", len(time_3s), np.mean(time_3s), np.sum(time_3s))
        print("polyak update: ", len(time_4s), np.mean(time_4s), np.sum(time_4s))
        # with open("pretrain_time.txt", "a") as f:
        #     # save the total time
        #     f.write(f"{np.sum(time_sample)}, "
        #             f"{np.sum(time_critic)}, "
        #             f"{np.sum(time_1s)}, "
        #             f"{np.sum(time_2s)}, "
        #             f"{np.sum(time_3s)}, "
        #             f"{np.sum(time_4s)}\n")

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/reconstruction_loss", np.mean(reconstruction_losses))

    def calculate_reconstruction_loss(self, observations):
        encoded_feature = self.actor.features_extractor(observations)
        l_feature = encoded_feature[:, : int(self.actor.features_extractor.features_dim / 2)]
        r_feature = encoded_feature[:, int(self.actor.features_extractor.features_dim / 2) :]
        marker_flow_obs = observations["marker_flow"]
        # (batch_num, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v))
        l_marker_initial_pos = marker_flow_obs[:, 0, 0, ...]
        r_marker_initial_pos = marker_flow_obs[:, 1, 0, ...]
        l_reconstructed_obs = self.policy.pointnet_decoder(l_marker_initial_pos, l_feature)
        # (batch_num, 128 (marker_num), 2 (u, v))
        r_reconstructed_obs = self.policy.pointnet_decoder(r_marker_initial_pos, r_feature)
        reconstruction_loss = self.reconstruction_loss_fn(
            l_reconstructed_obs, marker_flow_obs[:, 0, 1, ...]
        ) + self.reconstruction_loss_fn(r_reconstructed_obs, marker_flow_obs[:, 1, 1, ...])
        # two losses are added together, so it will be larger than pretraining time
        return reconstruction_loss

    def learn(
        self: CustomTD3Self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TD3",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> CustomTD3Self:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
