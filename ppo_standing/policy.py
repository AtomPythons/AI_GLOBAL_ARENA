"""PPO policy with a bounded mean for normalized joint commands."""

import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BoundedMeanPolicy(ActorCriticPolicy):
    """Keep the Gaussian action mean inside the physical action interval."""

    def _get_action_dist_from_latent(self, latent_pi):
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError("BoundedMeanPolicy requires Gaussian Box actions")
        mean_actions = torch.tanh(self.action_net(latent_pi))
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
