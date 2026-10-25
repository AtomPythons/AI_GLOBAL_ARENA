"""Finite static/sliding foot friction for the position-based arena.

The legacy contact rule only damps horizontal velocity. This module keeps
that rule untouched and supplies an opt-in foot point whose tangential
position correction is bounded by the floor's normal correction.
"""

import math

import ai_arena_physics_base as physics
from train_q_learning import Arena


class FrictionFootPoint(physics.Point):
    """Foot contact with finite static and sliding correction budgets."""

    def __init__(self, name, x, y, radius, static_mu, kinetic_mu):
        super().__init__(name, x, y, radius)
        if (not math.isfinite(static_mu) or not math.isfinite(kinetic_mu)
                or static_mu < kinetic_mu or kinetic_mu < 0):
            raise ValueError("Require finite static_mu >= kinetic_mu >= 0")
        self.static_mu = static_mu
        self.kinetic_mu = kinetic_mu
        self._anchor_x = self.position.x
        self._normal_correction = 0.0
        self._tangent_correction = 0.0

    def reset_contact_state(self):
        super().reset_contact_state()
        self._anchor_x = self.position.x
        self._normal_correction = 0.0
        self._tangent_correction = 0.0

    def solve_world_collision(self):
        if self.position.x - self.radius < physics.LEFT_WALL_X:
            self.position.x = physics.LEFT_WALL_X + self.radius
            self.velocity.x = max(0, -self.velocity.x * 0.2)
        if self.position.x + self.radius > physics.RIGHT_WALL_X:
            self.position.x = physics.RIGHT_WALL_X - self.radius
            self.velocity.x = min(0, -self.velocity.x * 0.2)
        if self.position.y - self.radius < 0:
            self.position.y = self.radius
            self.velocity.y = max(0, -self.velocity.y * 0.2)
        if self.position.y + self.radius < physics.FLOOR_Y:
            return
        penetration = max(0.0, self.position.y + self.radius - physics.FLOOR_Y)
        self.position.y = physics.FLOOR_Y - self.radius
        self.is_touching_floor = True
        self._normal_correction += penetration
        if self.velocity.y > 0:
            self.velocity.y *= -0.12
        drift = self.position.x - self._anchor_x
        static_budget = max(0.0, self.static_mu * self._normal_correction
                            - self._tangent_correction)
        kinetic_budget = max(0.0, self.kinetic_mu * self._normal_correction
                             - self._tangent_correction)
        if abs(drift) <= static_budget:
            correction = abs(drift)
        else:
            correction = min(abs(drift), kinetic_budget)
        self.position.x -= math.copysign(correction, drift)
        self._tangent_correction += correction
        self.velocity.x = ((self.position.x - self._anchor_x)
                           * physics.FPS * physics.SUBSTEPS)


class FiniteFrictionArena(Arena):
    """Original five-torque physics with opt-in finite foot-floor friction."""

    def __init__(self, episode_seconds=10.0, static_mu=0.2, kinetic_mu=0.1):
        self.static_mu = static_mu
        self.kinetic_mu = kinetic_mu
        super().__init__(episode_seconds=episode_seconds, posture_assist=False)

    def reset(self, seed=None):
        super().reset(seed)
        replacements = {}
        for index, point in enumerate(self.points):
            if point.name not in ("Left Foot", "Right Foot"):
                continue
            foot = FrictionFootPoint(point.name, point.position.x,
                                     point.position.y, point.radius,
                                     self.static_mu, self.kinetic_mu)
            foot.velocity = point.velocity.copy()
            foot.acceleration = point.acceleration.copy()
            foot.is_touching_floor = point.is_touching_floor
            self.points[index] = foot
            replacements[point] = foot
        for bone in self.bones:
            bone.point_a = replacements.get(bone.point_a, bone.point_a)
            bone.point_b = replacements.get(bone.point_b, bone.point_b)
        self.anatomy = physics.AnatomyConstraints(self.points, self.bones)
        self.metrics = physics.get_body_metrics(self.points)
        self.reward = physics.calculate_reward(self.metrics)
