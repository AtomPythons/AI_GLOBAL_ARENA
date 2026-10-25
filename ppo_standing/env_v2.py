"""Standing task with a measured, reasonably narrow foot stance."""

import numpy as np

import ai_arena_physics_base as physics
from .env import PPOStandingEnv, REWARD_FIELDS


STANCE_MIN = 35.0
STANCE_MAX = 120.0
SPREAD_PENALTY_START = 95.0
EXTRA_REWARD_FIELDS = ("stance_width", "foot_slip", "extra_effort")
REWARD_FIELDS_V2 = REWARD_FIELDS + EXTRA_REWARD_FIELDS


class PPOStableStanceEnv(PPOStandingEnv):
    """Same physics/actions/observations, with a longer-term stance signal."""

    def __init__(self, episode_seconds=10.0, control_frames=4):
        super().__init__(episode_seconds=episode_seconds,
                         control_frames=control_frames)
        self._stable_stance_frames = 0
        self._max_foot_span = 0.0
        self._extra_reward_components = dict.fromkeys(EXTRA_REWARD_FIELDS, 0.0)

    def _foot_state(self):
        body = {point.name: point for point in self.arena.points}
        left, right = body["Left Foot"], body["Right Foot"]
        span = abs(right.position.x - left.position.x)
        touching = [point for point in (left, right)
                    if point.is_touching_floor]
        mean_slip = (sum(abs(point.velocity.x) for point in touching)
                     / len(touching) if touching else 0.0)
        return span, mean_slip

    def _augment_info(self, info):
        span, slip = self._foot_state()
        frames = max(self._elapsed_frames, 1)
        stable_fraction = self._stable_stance_frames / frames
        info.update({
            "foot_span": span,
            "max_foot_span": self._max_foot_span,
            "foot_slip_speed": slip,
            "stable_stance_fraction": stable_fraction,
            "standing_success": (info["standing_success"]
                                 and stable_fraction >= 0.7),
        })
        info["episode_reward_components"].update(
            self._extra_reward_components)
        return info

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._stable_stance_frames = 0
        self._max_foot_span = 0.0
        self._extra_reward_components = dict.fromkeys(EXTRA_REWARD_FIELDS, 0.0)
        return observation, self._augment_info(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        span, slip = self._foot_state()
        frames = info["frames_this_step"]
        self._max_foot_span = max(self._max_foot_span, span)
        self._stable_stance_frames += frames * (STANCE_MIN <= span <= STANCE_MAX)
        duration = frames / physics.FPS
        spread = max(0.0, (span - SPREAD_PENALTY_START) / 80)
        collapse = max(0.0, (STANCE_MIN - span) / STANCE_MIN)
        components = {
            "stance_width": -(spread * spread + 0.5 * collapse * collapse)
                            * duration,
            "foot_slip": -0.15 * min(slip / 100, 1.0) * duration,
            "extra_effort": -0.10 * float(np.mean(np.abs(action))) * duration,
        }
        for name, value in components.items():
            self._extra_reward_components[name] += value
        info["reward_components"].update(components)
        self._augment_info(info)
        return (observation, reward + sum(components.values()),
                terminated, truncated, info)
