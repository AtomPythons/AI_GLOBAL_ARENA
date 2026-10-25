"""Standing task that accepts a compact, feet-together stance."""

import ai_arena_physics_base as physics
from .env import PPOStandingEnv
from .env_v3 import PPOControlledStanceEnv, REWARD_FIELDS_V3


REWARD_FIELDS_V4 = REWARD_FIELDS_V3 + ("compact_stance",)


class PPOCompactStanceEnv(PPOControlledStanceEnv):
    """Reward a narrow stance when it remains upright and foot supported."""

    def __init__(self, episode_seconds=10.0, control_frames=4):
        super().__init__(episode_seconds=episode_seconds,
                         control_frames=control_frames)
        self._compact_stance_frames = 0
        self._compact_stance_reward = 0.0

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._compact_stance_frames = 0
        self._compact_stance_reward = 0.0
        info["episode_reward_components"]["compact_stance"] = 0.0
        info["stable_stance_fraction"] = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        span = info["foot_span"]
        duration = info["frames_this_step"] / physics.FPS
        # V2/V3 penalized feet closer than 35/40 px. Cancel both penalties,
        # and give the same bonus that V3 gives for a 40-90 px stance.
        old_narrow = max(0.0, (40.0 - span) / 40.0)
        old_collapse = max(0.0, (35.0 - span) / 35.0)
        component = (0.5 * old_narrow**2 + 0.5 * old_collapse**2
                     + 0.25 * float(span < 40.0)) * duration
        self._compact_stance_reward += component
        self._compact_stance_frames += info["frames_this_step"] * (span <= 120.0)
        stable_fraction = (self._compact_stance_frames
                           / max(self._elapsed_frames, 1))
        info["stable_stance_fraction"] = stable_fraction
        info["standing_success"] = (
            PPOStandingEnv._info(self)["standing_success"]
            and stable_fraction >= 0.7 and span <= 120.0)
        info["reward_components"]["compact_stance"] = component
        info["episode_reward_components"]["compact_stance"] = (
            self._compact_stance_reward)
        return observation, reward + component, terminated, truncated, info
