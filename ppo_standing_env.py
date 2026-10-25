"""Headless Gymnasium environment for unassisted standing experiments."""

import math

import gymnasium as gym
import numpy as np

import ai_arena_physics_base as physics
from train_q_learning import ACTION_FRAMES, Arena


OBS_POINT_NAMES = (
    "Head", "Chest", "Left Elbow", "Left Hand", "Right Elbow",
    "Right Hand", "Left Knee", "Left Foot", "Right Knee", "Right Foot",
)
OBSERVATION_FIELDS = tuple(
    f"{name}.{field}"
    for name in OBS_POINT_NAMES
    for field in ("relative_x", "relative_y", "relative_vx", "relative_vy")
) + (
    "pelvis_vx", "pelvis_vy", "pelvis_height", "pelvis_support_offset",
    "torso_lean", "torso_angular_velocity", "left_foot_contact",
    "right_foot_contact",
)
REWARD_FIELDS = (
    "survival", "height", "upright", "feet", "other_support",
    "wall", "effort", "action_change", "fall",
)
FOOT_NAMES = frozenset(("Left Foot", "Right Foot"))


def standing_pose_metrics(points, body_metrics):
    """Physical standing criteria independent of the reward calculation."""
    foot_contact = bool(body_metrics["left_foot_contact"]
                        or body_metrics["right_foot_contact"])
    other_floor_contact = any(
        point.is_touching_floor and point.name not in FOOT_NAMES
        for point in points)
    wall_contact = any(
        point.position.x <= physics.LEFT_WALL_X + point.radius + 0.5
        or point.position.x >= physics.RIGHT_WALL_X - point.radius - 0.5
        for point in points)
    upright = (not body_metrics["is_fallen"]
               and body_metrics["pelvis_height"] >= 90
               and body_metrics["torso_angle"] <= 35)
    return {
        "foot_contact": foot_contact,
        "other_floor_contact": other_floor_contact,
        "wall_contact": wall_contact,
        "upright": upright,
        "valid_standing": (upright and foot_contact
                           and not other_floor_contact and not wall_contact),
    }


def standing_reward_components(body_metrics, pose_metrics, action,
                               previous_action):
    """Reward for one physics frame, with a one-time terminal fall cost."""
    fallen = bool(body_metrics["is_fallen"])
    dt = 1 / physics.FPS
    height_quality = max(0.0, min(1.0,
                                  (body_metrics["pelvis_height"] - 80) / 50))
    upright_quality = max(0.0, min(1.0,
                                   1 - body_metrics["torso_angle"] / 45))
    return {
        "survival": 0.0 if fallen else dt,
        "height": 0.0 if fallen else 0.15 * height_quality * dt,
        "upright": 0.0 if fallen else 0.15 * upright_quality * dt,
        "feet": 0.25 * dt if pose_metrics["foot_contact"] and not fallen else 0.0,
        "other_support": -1.0 * dt if pose_metrics["other_floor_contact"] else 0.0,
        "wall": -1.0 * dt if pose_metrics["wall_contact"] else 0.0,
        "effort": -0.03 * float(np.mean(np.square(action))) * dt,
        "action_change": -0.02 * float(np.mean(
            np.square(action - previous_action))) * dt,
        "fall": -0.25 if fallen else 0.0,
    }


class PPOStandingEnv(gym.Env):
    """Five bounded joint commands, without posture or spring assistance."""

    metadata = {"render_modes": []}

    def __init__(self, episode_seconds=3.0, control_frames=ACTION_FRAMES):
        super().__init__()
        if episode_seconds <= 0 or not isinstance(control_frames, int) \
                or control_frames < 1:
            raise ValueError("Episode duration and control frames must be positive")
        self.max_frames = round(episode_seconds * physics.FPS)
        if self.max_frames < 1:
            raise ValueError("Episode must contain at least one physics frame")
        self.control_frames = control_frames
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(physics.JOINT_ACTUATORS),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0,
            shape=(len(OBSERVATION_FIELDS),), dtype=np.float32)
        self.arena = Arena(episode_seconds=episode_seconds, posture_assist=False)
        self._elapsed_frames = 0
        self._standing_frames = 0
        self._done = False
        self._arena_seed = None
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._reward_components = dict.fromkeys(REWARD_FIELDS, 0.0)
        self._pose_counts = dict.fromkeys(
            ("foot_contact", "other_floor_contact", "wall_contact",
             "upright", "valid_standing"), 0)

    def _observation(self):
        body = {point.name: point for point in self.arena.points}
        pelvis = body["Pelvis"]
        chest = body["Chest"]
        values = []
        for name in OBS_POINT_NAMES:
            point = body[name]
            relative_position = point.position - pelvis.position
            relative_velocity = point.velocity - pelvis.velocity
            values.extend((relative_position.x / 200,
                           relative_position.y / 200,
                           relative_velocity.x / 500,
                           relative_velocity.y / 500))
        feet_x = (body["Left Foot"].position.x
                  + body["Right Foot"].position.x) / 2
        torso = pelvis.position - chest.position
        angular_velocity = torso.cross(
            pelvis.velocity - chest.velocity) / max(torso.length_squared(), 1)
        metrics = self.arena.metrics
        values.extend((
            pelvis.velocity.x / 500,
            pelvis.velocity.y / 500,
            metrics["pelvis_height"] / 200,
            (pelvis.position.x - feet_x) / 100,
            math.atan2(torso.x, torso.y) / math.pi,
            angular_velocity / 10,
            float(metrics["left_foot_contact"]),
            float(metrics["right_foot_contact"]),
        ))
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Physics produced a non-finite observation")
        return np.clip(np.asarray(values, dtype=np.float32), -5, 5)

    def _info(self, frames_this_step=0):
        metrics = self.arena.metrics
        pose = standing_pose_metrics(self.arena.points, metrics)
        frames = max(self._elapsed_frames, 1)
        return {
            "arena_seed": self._arena_seed,
            "elapsed_seconds": self._elapsed_frames / physics.FPS,
            "standing_seconds": self._standing_frames / physics.FPS,
            "frames_this_step": frames_this_step,
            "is_fallen": bool(metrics["is_fallen"]),
            "pelvis_height": float(metrics["pelvis_height"]),
            "torso_angle": float(metrics["torso_angle"]),
            "foot_contact": pose["foot_contact"],
            "other_floor_contact": pose["other_floor_contact"],
            "wall_contact": pose["wall_contact"],
            "upright": pose["upright"],
            "valid_standing": pose["valid_standing"],
            "foot_contact_fraction": self._pose_counts["foot_contact"] / frames,
            "other_support_fraction": (
                self._pose_counts["other_floor_contact"] / frames),
            "wall_contact_fraction": self._pose_counts["wall_contact"] / frames,
            "upright_fraction": self._pose_counts["upright"] / frames,
            "valid_standing_fraction": (
                self._pose_counts["valid_standing"] / frames),
            "standing_success": (
                self._done and not metrics["is_fallen"]
                and self._elapsed_frames >= self.max_frames
                and self._pose_counts["valid_standing"] / frames >= 0.7
                and self._pose_counts["other_floor_contact"] / frames <= 0.05
                and self._pose_counts["wall_contact"] / frames <= 0.05),
            "episode_reward_components": self._reward_components.copy(),
            "posture_assist": False,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "arena_seed" in options:
            arena_seed = options["arena_seed"]
            if (not isinstance(arena_seed, (int, np.integer))
                    or arena_seed < 0):
                raise ValueError("arena_seed must be a non-negative integer")
            self._arena_seed = int(arena_seed)
        else:
            self._arena_seed = int(self.np_random.integers(0, 2**31))
        self.arena.reset(self._arena_seed)
        self.arena.posture_assist = False
        self._elapsed_frames = 0
        self._standing_frames = 0
        self._done = False
        self._last_action.fill(0)
        self._reward_components = dict.fromkeys(REWARD_FIELDS, 0.0)
        self._pose_counts = dict.fromkeys(self._pose_counts, 0)
        return self._observation(), self._info()

    def step(self, action):
        if self._done or self._arena_seed is None:
            raise RuntimeError("Call reset() before stepping or after an episode ends")
        try:
            action = np.asarray(action, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid joint action") from error
        if not self.action_space.contains(action):
            raise ValueError("Joint action must contain five finite values in [-1, 1]")
        joint_action = physics.JointTorqueAction(tuple(float(value)
                                                 for value in action))
        step_components = dict.fromkeys(REWARD_FIELDS, 0.0)
        frames_this_step = 0
        terminated = False
        for _ in range(min(self.control_frames,
                           self.max_frames - self._elapsed_frames)):
            self.arena.step(joint_action)
            self._elapsed_frames += 1
            frames_this_step += 1
            terminated = bool(self.arena.metrics["is_fallen"])
            pose = standing_pose_metrics(self.arena.points, self.arena.metrics)
            for name in self._pose_counts:
                self._pose_counts[name] += int(pose[name])
            frame_components = standing_reward_components(
                self.arena.metrics, pose, action, self._last_action)
            for name, value in frame_components.items():
                step_components[name] += value
                self._reward_components[name] += value
            if terminated:
                break
            self._standing_frames += 1
        self._last_action = action.copy()
        truncated = self._elapsed_frames >= self.max_frames and not terminated
        self._done = terminated or truncated
        info = self._info(frames_this_step)
        info["reward_components"] = step_components
        return (self._observation(), sum(step_components.values()),
                terminated, truncated, info)
