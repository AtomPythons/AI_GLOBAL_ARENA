"""Train a Q-learning controller for the Pygame humanoid."""

import argparse
from datetime import datetime
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import pygame

import ai_arena_physics_base as physics
from spring_control import (POSE_ACTIONS, SPRING_ACTIONS, PoseAction,
                            SpringAction, apply_pose_action,
                            apply_spring_action)


FRAME_DT = 1 / physics.FPS
ACTION_FRAMES = 4
EPISODE_SECONDS = 3.0
EVALUATION_SEEDS = tuple(range(100, 150))
MODEL_VERSION = 1
JOINT_MODEL_VERSION = 2
FACTORED_MODEL_VERSION = 3
MOTION_MODEL_VERSION = 4
SURVIVAL_MODEL_VERSION = 5
SPRING_MODEL_VERSION = 6
POSE_MODEL_VERSION = 7
PRIMITIVE_ACTIONS = tuple(physics.get_action_space()[1:])
MOTION_ACTIONS = PRIMITIVE_ACTIONS + (
    "head_up", "left_knee_up", "right_knee_up")
MOTION_FORCES = {
    "head_up": ("Head", (0, -1200)),
    "left_knee_up": ("Left Knee", (0, -1200)),
    "right_knee_up": ("Right Knee", (0, -1200)),
}
JOINT_ACTION_COUNT = 1 << len(PRIMITIVE_ACTIONS)
ACTION_ABBREVIATIONS = ("C^", "C<", "C>", "P^", "P<", "P>",
                        "LF<", "LF>", "RF<", "RF>")
MOTION_LABELS = ACTION_ABBREVIATIONS + ("H^", "LK^", "RK^")
OPPOSING_FORCE_PAIRS = ((1, 2), (4, 5), (6, 7), (8, 9))
STATUS_HEIGHT = 160
EVALUATION_INTERVAL = 500
NUMBER_KEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
               pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
               pygame.K_9, pygame.K_0, pygame.K_MINUS)


@lru_cache(maxsize=JOINT_ACTION_COUNT)
def decode_joint_action(mask):
    if not 0 <= mask < JOINT_ACTION_COUNT:
        raise ValueError("Invalid joint action mask")
    return tuple(index + 1 for index in range(len(PRIMITIVE_ACTIONS))
                 if mask & (1 << index))


def write_model(path, payload):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bucket(value, edges):
    return sum(value >= edge for edge in edges)


def encode_state(points, metrics):
    """Keep only balance features that can be visited repeatedly."""
    body = {point.name: point for point in points}
    chest = body["Chest"]
    pelvis = body["Pelvis"]
    feet_x = (body["Left Foot"].position.x
              + body["Right Foot"].position.x) / 2
    torso = pelvis.position - chest.position
    lean = math.degrees(math.atan2(torso.x, torso.y))
    contacts = (int(metrics["left_foot_contact"])
                + 2 * int(metrics["right_foot_contact"]))
    return (
        bucket(lean, (-45, -20, -8, 8, 20, 45)),
        bucket(pelvis.position.x - feet_x, (-30, -10, 10, 30)),
        bucket(pelvis.velocity.x, (-60, -15, 15, 60)),
        bucket(metrics["pelvis_height"], (65, 95, 120)),
        contacts,
    )


def encode_motion_state(points, metrics):
    """Quantize pose and motion, relative to the current foot support."""
    body = {point.name: point for point in points}
    head, chest, pelvis = (body[name] for name in ("Head", "Chest", "Pelvis"))
    feet = (body["Left Foot"], body["Right Foot"])
    support_x = (feet[0].position.x + feet[1].position.x) / 2
    torso = pelvis.position - chest.position
    lean = math.degrees(math.atan2(torso.x, torso.y))
    angular_velocity = (pelvis.velocity.x - chest.velocity.x) / max(
        torso.length(), 1) * 180 / math.pi
    contacts = (int(metrics["left_foot_contact"])
                + 2 * int(metrics["right_foot_contact"]))
    return (
        bucket(lean, (-30, -10, 10, 30)),
        bucket(angular_velocity, (-80, -20, 20, 80)),
        bucket(pelvis.position.x - support_x, (-25, -8, 8, 25)),
        bucket(pelvis.velocity.x, (-80, -20, 20, 80)),
        bucket(pelvis.velocity.y, (-60, 10, 80)),
        bucket(metrics["pelvis_height"], (75, 105, 135)),
        bucket(head.position.x - chest.position.x, (-18, -6, 6, 18)),
        contacts,
    )


def motion_features(state):
    """Share experience across similar poses without a giant joint table."""
    return ((0, state[0], state[1]),
            (1, state[2], state[3]),
            (2, state[4], state[5]),
            (3, state[0], state[2]),
            (4, state[5], state[7]),
            (5, state[6], state[1]))


def apply_motion_force(points, action):
    if action in MOTION_FORCES:
        name, force = MOTION_FORCES[action]
        next(point for point in points if point.name == name).apply_force(
            pygame.Vector2(force))
    else:
        physics.apply_discrete_action(points,
                                      physics.get_action_space().index(action))


def resolve_opposing_forces(mask, gains):
    """Keep the stronger useful direction when two forces cancel out."""
    for left, right in OPPOSING_FORCE_PAIRS:
        if mask & (1 << left) and mask & (1 << right):
            weaker = right if gains[left] >= gains[right] else left
            mask &= ~(1 << weaker)
    return mask


class Arena:
    def __init__(self, episode_seconds=EPISODE_SECONDS, posture_assist=True):
        self.episode_seconds = episode_seconds
        self.posture_assist = posture_assist
        self.episode = physics.EpisodeManager(episode_seconds)
        self.reset()

    def reset(self, seed=None):
        self.points, self.bones = physics.create_humanoid()
        self.episode.reset()
        if seed is not None:
            rng = random.Random(seed)
            offset = rng.uniform(-15, 15)
            for point in self.points:
                point.position.x += offset + rng.uniform(-4, 4)
                point.velocity.x = rng.uniform(-25, 25)
            body = {point.name: point for point in self.points}
            body["Chest"].position.x += rng.uniform(-10, 10)
            body["Head"].position.x += rng.uniform(-14, 14)
        self.metrics = physics.get_body_metrics(self.points)
        self.reward = physics.calculate_reward(self.metrics)

    def step(self, action, selected=None, keys=None, mouse_position=None,
             controller=None):
        """Use the same substeps and collision order as the physics demo."""
        sub_dt = FRAME_DT / physics.SUBSTEPS
        for _ in range(physics.SUBSTEPS):
            for point in self.points:
                point.reset_contact_state()
                point.apply_force(pygame.Vector2(0, physics.GRAVITY_Y))
            if self.posture_assist:
                physics.apply_posture_assist(self.points)
            if isinstance(action, PoseAction):
                apply_pose_action(self.points, action.mask)
            elif isinstance(action, SpringAction):
                apply_spring_action(self.points, action.mask)
            elif isinstance(action, int):
                physics.apply_discrete_action(self.points, action)
            else:
                for primitive in action:
                    if isinstance(primitive, str):
                        apply_motion_force(self.points, primitive)
                    else:
                        physics.apply_discrete_action(self.points, primitive)
            if controller is not None:
                controller.apply(self.points)
            if selected is not None and keys is not None:
                physics.apply_manual_force(selected, keys, mouse_position)
            previous = [point.position.copy() for point in self.points]
            for point in self.points:
                point.update(sub_dt)
            for _ in range(physics.CONSTRAINT_ITERATIONS):
                for bone in self.bones:
                    bone.solve()
                for point in self.points:
                    point.solve_world_collision()
            for point, old_position in zip(self.points, previous):
                point.velocity = (point.position - old_position) / sub_dt
                point.solve_world_collision()
        self.metrics = physics.get_body_metrics(self.points)
        self.reward = physics.calculate_reward(self.metrics)
        self.episode.update(FRAME_DT, self.metrics["is_fallen"])
        return self.reward


class QAgent:
    def __init__(self, alpha=0.15, gamma=0.97, epsilon=1.0):
        self.actions = physics.get_action_space()
        self.values = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes_trained = 0
        self.best_mean_reward = float("-inf")
        self.best_mean_alive = float("-inf")
        self.checkpoint_mean_reward = float("-inf")

    @property
    def action_count(self):
        return len(self.actions)

    def row(self, state):
        return self.values.setdefault(state, [0.0] * len(self.actions))

    def greedy_action(self, state):
        values = self.values.get(state)
        if values is None:
            return 0
        return max(range(len(values)), key=values.__getitem__)

    def choose_action(self, state, rng):
        if rng.random() < self.epsilon:
            return rng.randrange(len(self.actions))
        return self.greedy_action(state)

    def decode_action(self, action):
        return action

    def action_label(self, action):
        return self.actions[action]

    def encode_state(self, points, metrics):
        return encode_state(points, metrics)

    def frame_reward(self, arena):
        return arena.reward

    def transition_reward(self, total, frames):
        return total / frames

    def learn(self, state, action, reward, next_state, done):
        values = self.row(state)
        future = 0 if done else max(self.row(next_state))
        values[action] += self.alpha * (
            reward + self.gamma * future - values[action])

    def finish_episode(self):
        self.episodes_trained += 1
        self.epsilon = max(0.05, 1.0 * 0.9995 ** self.episodes_trained)

    def save(self, path):
        payload = {
            "version": MODEL_VERSION,
            "algorithm": "tabular_q_learning",
            "actions": self.actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "best_mean_reward": self.best_mean_reward,
            "best_mean_alive": self.best_mean_alive,
            "checkpoint_mean_reward": self.checkpoint_mean_reward,
            "episode_seconds": EPISODE_SECONDS,
            "action_frames": ACTION_FRAMES,
            "posture_assist": True,
            "q_table": {",".join(map(str, state)): values
                        for state, values in self.values.items()},
        }
        write_model(path, payload)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("version") == POSE_MODEL_VERSION:
            return PoseQAgent.from_payload(payload)
        if payload.get("version") == SPRING_MODEL_VERSION:
            return SpringQAgent.from_payload(payload)
        if payload.get("version") == SURVIVAL_MODEL_VERSION:
            return SurvivalQAgent.from_payload(payload)
        if payload.get("version") == FACTORED_MODEL_VERSION:
            return FactoredQAgent.from_payload(payload)
        if payload.get("version") == MOTION_MODEL_VERSION:
            return MotionQAgent.from_payload(payload)
        if payload.get("version") == JOINT_MODEL_VERSION:
            return JointQAgent.from_payload(payload)
        if payload.get("version") != MODEL_VERSION:
            raise ValueError("Unsupported model version")
        if payload.get("algorithm") != "tabular_q_learning":
            raise ValueError("Unsupported model algorithm")
        if payload.get("actions") != physics.get_action_space():
            raise ValueError("Model action space does not match the physics module")
        if payload.get("action_frames") != ACTION_FRAMES:
            raise ValueError("Model action interval does not match the viewer")
        agent = cls(payload["alpha"], payload["gamma"], payload["epsilon"])
        agent.episodes_trained = payload["episodes_trained"]
        agent.best_mean_reward = payload["best_mean_reward"]
        agent.best_mean_alive = payload.get("best_mean_alive", float("-inf"))
        agent.checkpoint_mean_reward = payload.get(
            "checkpoint_mean_reward", agent.best_mean_reward)
        agent.values = {
            tuple(map(int, key.split(","))): values
            for key, values in payload["q_table"].items()
        }
        return agent


class JointQAgent(QAgent):
    """Tabular Q-learning over every subset of the primitive forces."""

    @property
    def action_count(self):
        return JOINT_ACTION_COUNT

    def row(self, state):
        return self.values.setdefault(state, {})

    def greedy_action(self, state):
        values = self.values.get(state)
        if not values:
            return 0
        action = max(values, key=values.get)
        return action if values[action] > 0 else 0

    def choose_action(self, state, rng):
        if rng.random() >= self.epsilon:
            return self.greedy_action(state)
        if rng.random() < 0.1:
            return rng.randrange(JOINT_ACTION_COUNT)
        mask = 0
        for index in range(len(PRIMITIVE_ACTIONS)):
            if rng.random() < 0.25:
                mask |= 1 << index
        return mask

    def learn(self, state, action, reward, next_state, done):
        values = self.row(state)
        next_values = self.values.get(next_state, {})
        future = 0 if done else max((0, *next_values.values()))
        old_value = values.get(action, 0.0)
        values[action] = old_value + self.alpha * (
            reward + self.gamma * future - old_value)

    def decode_action(self, action):
        return decode_joint_action(action)

    def action_label(self, action):
        if action == 0:
            return "none"
        return " ".join(label for index, label in enumerate(ACTION_ABBREVIATIONS)
                        if action & (1 << index))

    def save(self, path):
        payload = {
            "version": JOINT_MODEL_VERSION,
            "algorithm": "tabular_joint_q_learning",
            "primitive_actions": PRIMITIVE_ACTIONS,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "best_mean_reward": self.best_mean_reward,
            "best_mean_alive": self.best_mean_alive,
            "checkpoint_mean_reward": self.checkpoint_mean_reward,
            "episode_seconds": EPISODE_SECONDS,
            "action_frames": ACTION_FRAMES,
            "posture_assist": True,
            "q_table": {",".join(map(str, state)):
                        {str(mask): value for mask, value in values.items()}
                        for state, values in self.values.items()},
        }
        write_model(path, payload)

    @classmethod
    def from_payload(cls, payload):
        if payload.get("algorithm") != "tabular_joint_q_learning":
            raise ValueError("Unsupported joint model algorithm")
        if tuple(payload.get("primitive_actions", ())) != PRIMITIVE_ACTIONS:
            raise ValueError("Model actions do not match the physics module")
        if payload.get("action_frames") != ACTION_FRAMES:
            raise ValueError("Model action interval does not match the viewer")
        agent = cls(payload["alpha"], payload["gamma"], payload["epsilon"])
        agent.episodes_trained = payload["episodes_trained"]
        agent.best_mean_reward = payload["best_mean_reward"]
        agent.best_mean_alive = payload.get("best_mean_alive", float("-inf"))
        agent.checkpoint_mean_reward = payload.get(
            "checkpoint_mean_reward", agent.best_mean_reward)
        agent.values = {
            tuple(map(int, state.split(","))):
            {int(mask): value for mask, value in values.items()}
            for state, values in payload["q_table"].items()
        }
        return agent

    @classmethod
    def from_legacy(cls, legacy):
        agent = cls(legacy.alpha, legacy.gamma)
        agent.epsilon = legacy.epsilon
        agent.episodes_trained = legacy.episodes_trained
        agent.best_mean_reward = legacy.best_mean_reward
        agent.best_mean_alive = legacy.best_mean_alive
        agent.checkpoint_mean_reward = legacy.checkpoint_mean_reward
        agent.values = {
            state: {0: values[0], **{
                1 << (index - 1): value
                for index, value in enumerate(values) if index > 0 and value != 0
            }}
            for state, values in legacy.values.items()
        }
        return agent


class FactoredQAgent(JointQAgent):
    """Approximate joint Q-values with one on/off factor per force."""

    def row(self, state):
        return self.values.setdefault(
            state, [[0.0, 0.0] for _ in PRIMITIVE_ACTIONS])

    def greedy_action(self, state):
        factors = self.values.get(state)
        if factors is None:
            return 0
        return sum(1 << index for index, (off, on) in enumerate(factors)
                   if on > off)

    def learn(self, state, action, reward, next_state, done):
        factors = self.row(state)
        chosen = tuple((action >> index) & 1
                       for index in range(len(PRIMITIVE_ACTIONS)))
        estimate = sum(factor[bit] for factor, bit
                       in zip(factors, chosen)) / len(factors)
        future = 0.0
        if not done:
            next_factors = self.row(next_state)
            future = sum(max(factor) for factor in next_factors) / len(next_factors)
        error = reward + self.gamma * future - estimate
        for factor, bit in zip(factors, chosen):
            factor[bit] += self.alpha * error

    def save(self, path):
        payload = {
            "version": FACTORED_MODEL_VERSION,
            "algorithm": "factored_q_learning",
            "primitive_actions": PRIMITIVE_ACTIONS,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "best_mean_reward": self.best_mean_reward,
            "best_mean_alive": self.best_mean_alive,
            "checkpoint_mean_reward": self.checkpoint_mean_reward,
            "episode_seconds": EPISODE_SECONDS,
            "action_frames": ACTION_FRAMES,
            "posture_assist": True,
            "q_table": {",".join(map(str, state)): factors
                        for state, factors in self.values.items()},
        }
        write_model(path, payload)

    @classmethod
    def from_payload(cls, payload):
        if payload.get("algorithm") != "factored_q_learning":
            raise ValueError("Unsupported factored model algorithm")
        if tuple(payload.get("primitive_actions", ())) != PRIMITIVE_ACTIONS:
            raise ValueError("Model actions do not match the physics module")
        if payload.get("action_frames") != ACTION_FRAMES:
            raise ValueError("Model action interval does not match the viewer")
        agent = cls(payload["alpha"], payload["gamma"], payload["epsilon"])
        agent.episodes_trained = payload["episodes_trained"]
        agent.best_mean_reward = payload["best_mean_reward"]
        agent.best_mean_alive = payload["best_mean_alive"]
        agent.checkpoint_mean_reward = payload["checkpoint_mean_reward"]
        agent.values = {
            tuple(map(int, state.split(","))): factors
            for state, factors in payload["q_table"].items()
        }
        if any(len(factors) != len(PRIMITIVE_ACTIONS)
               or any(len(factor) != 2 for factor in factors)
               for factors in agent.values.values()):
            raise ValueError("Invalid factored Q-table shape")
        return agent


class MotionQAgent(QAgent):
    """Feature-based Q-learning over simultaneous body forces."""

    model_version = MOTION_MODEL_VERSION
    algorithm = "motion_feature_q_learning"
    reward_mode = "posture"
    action_names = MOTION_ACTIONS

    def __init__(self, alpha=0.035, gamma=0.97, epsilon=1.0):
        super().__init__(alpha, gamma, epsilon)
        self.actions = self.action_names
        self.resolve_opposites = True

    @property
    def action_count(self):
        return 1 << len(self.actions)

    def encode_state(self, points, metrics):
        return encode_motion_state(points, metrics)

    def row(self, feature):
        return self.values.setdefault(
            feature, [[0.0, 0.0] for _ in self.actions])

    def scores(self, state):
        features = motion_features(state)
        rows = [self.values.get(feature) for feature in features]
        return [[sum(row[index][bit] for row in rows if row is not None)
                 / len(features) for bit in (0, 1)]
                for index in range(len(self.actions))]

    def greedy_action(self, state):
        gains = [on - off for off, on in self.scores(state)]
        mask = sum(1 << index for index, gain in enumerate(gains) if gain > 0)
        return (resolve_opposing_forces(mask, gains)
                if self.resolve_opposites else mask)

    def choose_action(self, state, rng):
        if rng.random() >= self.epsilon:
            return self.greedy_action(state)
        if rng.random() < 0.1:
            return rng.randrange(self.action_count)
        mask = 0
        for index in range(len(self.actions)):
            if rng.random() < 0.25:
                mask |= 1 << index
        return (resolve_opposing_forces(mask, [rng.random() for _ in self.actions])
                if self.resolve_opposites else mask)

    def decode_action(self, action):
        if not 0 <= action < self.action_count:
            raise ValueError("Invalid motion action mask")
        return tuple(name for index, name in enumerate(self.actions)
                     if action & (1 << index))

    def action_label(self, action):
        labels = [label for index, label in enumerate(MOTION_LABELS)
                  if action & (1 << index)]
        return " ".join(labels) if labels else "none"

    def learn(self, state, action, reward, next_state, done):
        features = motion_features(state)
        scores = self.scores(state)
        estimate = sum(scores[index][(action >> index) & 1]
                       for index in range(len(self.actions))) / len(self.actions)
        future = 0.0
        if not done:
            future = sum(max(pair) for pair in self.scores(next_state)) / len(
                self.actions)
        error = reward + self.gamma * future - estimate
        for feature in features:
            row = self.row(feature)
            for index in range(len(self.actions)):
                row[index][(action >> index) & 1] += self.alpha * error

    def save(self, path):
        payload = {
            "version": self.model_version,
            "algorithm": self.algorithm,
            "reward_mode": self.reward_mode,
            "actions": self.actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "best_mean_reward": self.best_mean_reward,
            "best_mean_alive": self.best_mean_alive,
            "checkpoint_mean_reward": self.checkpoint_mean_reward,
            "episode_seconds": EPISODE_SECONDS,
            "action_frames": ACTION_FRAMES,
            "posture_assist": True,
            "resolve_opposites": self.resolve_opposites,
            "q_table": {",".join(map(str, feature)): values
                        for feature, values in self.values.items()},
        }
        write_model(path, payload)

    @classmethod
    def from_payload(cls, payload):
        if payload.get("algorithm") != cls.algorithm:
            raise ValueError("Unsupported motion model algorithm")
        if payload.get("reward_mode", "posture") != cls.reward_mode:
            raise ValueError("Model reward mode does not match the algorithm")
        if tuple(payload.get("actions", ())) != cls.action_names:
            raise ValueError("Model actions do not match the physics module")
        if payload.get("action_frames") != ACTION_FRAMES:
            raise ValueError("Model action interval does not match the viewer")
        agent = cls(payload["alpha"], payload["gamma"], payload["epsilon"])
        agent.resolve_opposites = payload.get("resolve_opposites", False)
        agent.episodes_trained = payload["episodes_trained"]
        agent.best_mean_reward = payload["best_mean_reward"]
        agent.best_mean_alive = payload["best_mean_alive"]
        agent.checkpoint_mean_reward = payload["checkpoint_mean_reward"]
        agent.values = {
            tuple(map(int, key.split(","))): values
            for key, values in payload["q_table"].items()
        }
        if any(len(row) != len(cls.action_names)
               or any(len(pair) != 2 for pair in row)
               for row in agent.values.values()):
            raise ValueError("Invalid motion Q-table shape")
        return agent


class SurvivalQAgent(MotionQAgent):
    """Optimize the number of frames spent upright, not the posture score."""

    model_version = SURVIVAL_MODEL_VERSION
    algorithm = "survival_feature_q_learning"
    reward_mode = "survival"

    def frame_reward(self, arena):
        return -1.0 if arena.metrics["is_fallen"] else 1.0

    def transition_reward(self, total, frames):
        return total


class SpringQAgent(SurvivalQAgent):
    """Learn which pose-dependent force groups to enable together."""

    model_version = SPRING_MODEL_VERSION
    algorithm = "spring_feature_q_learning"
    action_names = SPRING_ACTIONS

    def __init__(self, alpha=0.035, gamma=0.97, epsilon=1.0):
        super().__init__(alpha, gamma, epsilon)
        self.resolve_opposites = False

    def choose_action(self, state, rng):
        if rng.random() < self.epsilon:
            return rng.randrange(self.action_count)
        return self.greedy_action(state)

    def decode_action(self, action):
        if not 0 <= action < self.action_count:
            raise ValueError("Invalid spring action mask")
        return SpringAction(action)

    def action_label(self, action):
        labels = ("P", "C", "H", "LL", "RL")
        active = [label for index, label in enumerate(labels)
                  if action & (1 << index)]
        return " ".join(active) if active else "none"


class PoseQAgent(SpringQAgent):
    """Select mirrored leg and arm feedback without altering old models."""

    model_version = POSE_MODEL_VERSION
    algorithm = "pose_feature_q_learning"
    action_names = POSE_ACTIONS

    def decode_action(self, action):
        if not 0 <= action < self.action_count:
            raise ValueError("Invalid pose action mask")
        return PoseAction(action)

    def action_label(self, action):
        labels = ("P", "C", "H", "LL", "RL", "LA", "RA")
        active = [label for index, label in enumerate(labels)
                  if action & (1 << index)]
        return " ".join(active) if active else "none"


def run_episode(agent, rng, training, seed):
    arena = Arena()
    arena.reset(seed)
    total_reward = 0.0
    total_posture_reward = 0.0
    standing_score_total = 0
    frame_count = 0
    while not arena.episode.is_episode_over:
        state = agent.encode_state(arena.points, arena.metrics)
        action = agent.choose_action(state, rng) if training else agent.greedy_action(state)
        reward = 0.0
        posture_reward = 0.0
        frames = 0
        for _ in range(ACTION_FRAMES):
            arena.step(agent.decode_action(action))
            reward += agent.frame_reward(arena)
            posture_reward += arena.reward
            standing_score_total += arena.metrics["standing_score"]
            frame_count += 1
            frames += 1
            if arena.episode.is_episode_over:
                break
        reward = agent.transition_reward(reward, frames)
        total_reward += reward
        total_posture_reward += posture_reward / frames
        if training:
            next_state = agent.encode_state(arena.points, arena.metrics)
            agent.learn(state, action, reward, next_state,
                        arena.episode.is_episode_over)
    if training:
        agent.finish_episode()
    arena.total_posture_reward = total_posture_reward
    arena.mean_standing_score = standing_score_total / frame_count
    return total_reward, arena.episode.time_alive, arena


def evaluate(agent):
    rng = random.Random(0)
    results = [run_episode(agent, rng, False, seed)
               for seed in EVALUATION_SEEDS]
    return (sum(result[0] for result in results) / len(results),
            sum(result[1] for result in results) / len(results),
            sum(result[2].total_posture_reward for result in results)
            / len(results),
            sum(result[2].mean_standing_score for result in results)
            / len(results))


def draw_overlay(screen, font, lines):
    height = len(lines) * 19 + 9
    top = physics.SCREEN_HEIGHT + 4
    pygame.draw.rect(screen, (8, 12, 18),
                     (4, top, physics.SCREEN_WIDTH - 8, height))
    for index, line in enumerate(lines):
        screen.blit(font.render(line, True, physics.TEXT_COLOR),
                    (10, top + 4 + index * 19))
    pygame.display.flip()


def draw_arena(screen, font, small_font, arena, selected, show_labels, lines):
    physics.draw_scene(screen, font, small_font, arena.points, arena.bones,
                       selected, show_labels, arena.metrics, arena.reward,
                       arena.episode, arena.posture_assist)
    draw_overlay(screen, font, lines)


def run_demo(screen, font, small_font, agent, model_name, allow_continue=False,
             total_episodes=None, session_start=0):
    """Play a greedy policy with the original demo's inspection controls."""
    clock = pygame.time.Clock()
    arena = Arena()
    selected = arena.points[0]
    show_labels = False
    dragging = False
    model_enabled = True
    action = 0
    frames_left = 0
    seed = None
    while True:
        clock.tick(physics.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if allow_continue and event.key == pygame.K_c:
                    return "continue"
                if event.key in NUMBER_KEYS:
                    selected = arena.points[NUMBER_KEYS.index(event.key)]
                elif event.key == pygame.K_l:
                    show_labels = not show_labels
                elif event.key == pygame.K_p:
                    arena.posture_assist = not arena.posture_assist
                elif event.key == pygame.K_m:
                    model_enabled = not model_enabled
                    frames_left = 0
                elif event.key in (pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_n:
                        arena.episode.next_episode()
                        seed = random.randrange(1_000_000)
                    else:
                        seed = None
                    arena.reset(seed)
                    selected = arena.points[0]
                    dragging = False
                    frames_left = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = physics.find_point_at_mouse(arena.points, event.pos)
                if clicked is not None:
                    selected = clicked
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

        if not arena.episode.is_episode_over:
            if frames_left == 0:
                action = (agent.greedy_action(agent.encode_state(arena.points, arena.metrics))
                          if model_enabled else 0)
                frames_left = ACTION_FRAMES
            arena.step(agent.decode_action(action), selected,
                       pygame.key.get_pressed(),
                       pygame.mouse.get_pos() if dragging else None)
            frames_left -= 1

        best_alive = (f"{agent.best_mean_alive:.2f}s"
                      if math.isfinite(agent.best_mean_alive) else "n/a")
        trained = agent.episodes_trained if total_episodes is None else total_episodes
        progress = (f"Trained: {trained}"
                    if total_episodes is None else
                    f"Total episodes: {trained}  "
                    f"This session: {trained - session_start}")
        lines = [f"Model: {model_name}", progress,
                 f"Best standing: {best_alive}  "
                 f"Training return: {agent.checkpoint_mean_reward:.1f}",
                 f"Control: {'model' if model_enabled else 'manual'}  "
                 f"Forces: {agent.action_label(action)}",
                 "R replay  N new  M model  P assist  L labels",
                 "Arrows/mouse: move point  1-0/-: select"]
        if allow_continue:
            lines.append("C: train next batch  C/P/L/RF: body parts")
        draw_arena(screen, font, small_font, arena, selected, show_labels, lines)


class TrainingSession:
    def __init__(self, started_at, best_path, last_path):
        self.started_at = started_at
        self.best_path = best_path
        self.last_path = last_path
        self.best_mean_alive = float("-inf")
        self.best_mean_reward = float("-inf")

    def episodes_completed(self, agent):
        return agent.episodes_trained - self.started_at

    def save_best_if_improved(self, agent, mean_alive, mean_reward):
        improved = (mean_alive > self.best_mean_alive + 1e-9
                    or (abs(mean_alive - self.best_mean_alive) <= 1e-9
                        and mean_reward > self.best_mean_reward))
        if not improved:
            return False
        self.best_mean_alive = mean_alive
        self.best_mean_reward = mean_reward
        agent.save(self.best_path)
        return True

    def save_last(self, agent):
        if self.episodes_completed(agent) == 0:
            return
        agent.save(self.last_path)
        if not self.best_path.exists():
            agent.save(self.best_path)


def train_batch(agent, count, screen, font, small_font, session, rng):
    started = time.monotonic()
    last_draw = 0.0
    mean_reward = 0.0
    mean_alive = 0.0
    completed = 0
    for index in range(count):
        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
        reward, alive, arena = run_episode(
            agent, rng, True, rng.randrange(1_000_000_000))
        completed += 1
        mean_reward += (reward - mean_reward) / completed
        mean_alive += (alive - mean_alive) / completed
        now = time.monotonic()
        if (((index + 1) % EVALUATION_INTERVAL == 0)
                or index == count - 1):
            (evaluation_reward, evaluation_alive, evaluation_posture,
             evaluation_standing_score) = evaluate(agent)
            agent.best_mean_reward = max(agent.best_mean_reward,
                                         evaluation_reward)
            if (evaluation_alive > agent.best_mean_alive + 1e-9
                    or (abs(evaluation_alive - agent.best_mean_alive) <= 1e-9
                        and evaluation_reward > agent.checkpoint_mean_reward)):
                agent.best_mean_alive = evaluation_alive
                agent.checkpoint_mean_reward = evaluation_reward
            if session.save_best_if_improved(
                    agent, evaluation_alive, evaluation_reward):
                print(f"Saved best model of session: {session.best_path}",
                      flush=True)
            print(f"Evaluated after {agent.episodes_trained} episodes: "
                  f"this session {session.episodes_completed(agent)}, "
                  f"mean reward {evaluation_reward:.1f}, "
                  f"last episode reward {reward:.1f}, "
                  f"standing {evaluation_alive:.3f}s, "
                  f"best standing {agent.best_mean_alive:.3f}s, "
                  f"mean standing score {evaluation_standing_score:.2f}/5, "
                  f"posture return {evaluation_posture:.1f}, "
                  f"peak return {agent.best_mean_reward:.1f}", flush=True)
        if screen is None:
            if (index + 1) % min(1000, count) == 0:
                print(f"Episodes: {agent.episodes_trained}  "
                      f"This session: {session.episodes_completed(agent)}  "
                      f"Batch: {index + 1}/{count}  "
                      f"Mean training return: {mean_reward:.1f}  "
                      f"Last episode reward: {reward:.1f}  "
                      f"Standing score: {arena.mean_standing_score:.2f}/5 "
                      f"(episode mean)", flush=True)
        elif now - last_draw >= 0.15 or index == count - 1:
            rate = completed / max(now - started, 0.001)
            remaining = (count - completed) / rate
            best_alive = (f"{agent.best_mean_alive:.2f}s"
                          if math.isfinite(agent.best_mean_alive)
                          else "pending")
            best_reward = (f"{agent.best_mean_reward:.1f}"
                           if math.isfinite(agent.best_mean_reward)
                           else "pending")
            lines = [f"TRAINING  {index + 1}/{count} in this batch",
                     f"Total episodes: {agent.episodes_trained}  "
                     f"Session: {session.episodes_completed(agent)}  "
                     f"ETA: {remaining:.0f}s",
                     f"Best standing: {best_alive}  "
                     f"Batch: {mean_alive:.2f}s",
                     f"Peak return: {best_reward}  "
                     f"Batch: {mean_reward:.1f}",
                     f"Last episode reward: {reward:.1f}  "
                     f"Standing score: {arena.mean_standing_score:.2f}/5 "
                     f"(episode mean)",
                     f"Exploration: {agent.epsilon:.2f}  "
                     f"Actions: {agent.action_count}"]
            draw_arena(screen, font, small_font, arena, arena.points[0],
                       False, lines)
            last_draw = now
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Continue from a saved model")
    parser.add_argument("--agent", choices=("spring", "direct", "pose"),
                        help="Controller for a new run (default: spring)")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--batches", type=int, default=1,
                        help="Number of batches in headless mode")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batches < 1:
        parser.error("Batch size and batch count must be positive")
    if args.model and args.agent:
        parser.error("--agent applies only to a new run without --model")

    rng = random.Random()
    saved = QAgent.load(args.model) if args.model else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_prefix = f"ai_model_{timestamp}"
    model_dir = Path(__file__).parent
    if saved is None:
        agent_type = {"spring": SpringQAgent, "direct": SurvivalQAgent,
                      "pose": PoseQAgent}[args.agent or "spring"]
        agent = agent_type()
    elif type(saved) is QAgent:
        agent = JointQAgent.from_legacy(saved)
        print(f"Imported single-action values from {args.model}.", flush=True)
    else:
        agent = saved
    session = TrainingSession(
        agent.episodes_trained,
        model_dir / f"{model_prefix}_best_model_of_session.json",
        model_dir / f"{model_prefix}_last_model_of_session.json")
    if args.model:
        print(f"Continuing {args.model} from {session.started_at} episodes; "
              f"new checkpoints will be saved separately.", flush=True)
    print(f"Available action combinations: {agent.action_count}", flush=True)
    screen = font = small_font = None
    if not args.headless:
        pygame.init()
        screen = pygame.display.set_mode(
            (physics.SCREEN_WIDTH, physics.SCREEN_HEIGHT + STATUS_HEIGHT))
        pygame.display.set_caption("AI Arena - Q-learning")
        font = pygame.font.SysFont(None, 20)
        small_font = pygame.font.SysFont(None, 16)
    try:
        batch_numbers = range(args.batches) if args.headless else itertools.count()
        for _ in batch_numbers:
            if not train_batch(agent, args.batch_size, screen, font,
                               small_font, session, rng):
                break
            if screen is not None:
                best = QAgent.load(session.best_path)
                if run_demo(screen, font, small_font, best,
                            session.best_path.name, allow_continue=True,
                            total_episodes=agent.episodes_trained,
                            session_start=session.started_at) == "quit":
                    break
    except KeyboardInterrupt:
        print("Stopping training...", flush=True)
    finally:
        try:
            session.save_last(agent)
            if session.last_path.exists():
                print(f"Saved last model of session: {session.last_path}",
                      flush=True)
                print(f"Best model of session: {session.best_path}", flush=True)
        finally:
            if not args.headless:
                pygame.quit()


if __name__ == "__main__":
    main()
