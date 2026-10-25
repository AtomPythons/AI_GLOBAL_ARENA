"""Standing task with a stronger incentive to keep feet under the body."""

import ai_arena_physics_base as physics
from ppo_standing_env_v2 import PPOStableStanceEnv, REWARD_FIELDS_V2


REWARD_FIELDS_V3 = REWARD_FIELDS_V2 + ("stance_control",)


class PPOControlledStanceEnv(PPOStableStanceEnv):
    """Use the V2 task and add a denser cost for legs spreading apart."""

    def __init__(self, episode_seconds=10.0, control_frames=4):
        super().__init__(episode_seconds=episode_seconds,
                         control_frames=control_frames)
        self._stance_control_reward = 0.0

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._stance_control_reward = 0.0
        info["episode_reward_components"]["stance_control"] = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        span = info["foot_span"]
        seconds = info["frames_this_step"] / physics.FPS
        wide = max(0.0, (span - 90.0) / 80.0)
        narrow = max(0.0, (40.0 - span) / 40.0)
        stable = 40.0 <= span <= 90.0
        component = (0.25 * float(stable) - 4.0 * wide**2
                     - 0.5 * narrow**2) * seconds
        self._stance_control_reward += component
        info["reward_components"]["stance_control"] = component
        info["episode_reward_components"]["stance_control"] = (
            self._stance_control_reward)
        if truncated:
            info["standing_success"] = (info["standing_success"]
                                        and 35.0 <= span <= 120.0)
        return observation, reward + component, terminated, truncated, info
