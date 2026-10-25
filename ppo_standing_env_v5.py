"""Compact-stance PPO task with finite foot-floor friction."""

from finite_foot_friction import FiniteFrictionArena
from ppo_standing_env_v4 import PPOCompactStanceEnv, REWARD_FIELDS_V4


REWARD_FIELDS_V5 = REWARD_FIELDS_V4


class PPOFiniteFrictionEnv(PPOCompactStanceEnv):
    """Keep V4 actions, observations and reward; change foot contact only."""

    def __init__(self, episode_seconds=10.0, control_frames=4,
                 static_mu=0.2, kinetic_mu=0.1):
        super().__init__(episode_seconds=episode_seconds,
                         control_frames=control_frames)
        self.arena = FiniteFrictionArena(
            episode_seconds=episode_seconds,
            static_mu=static_mu, kinetic_mu=kinetic_mu)
