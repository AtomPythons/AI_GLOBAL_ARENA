"""Pose-dependent force groups for the humanoid controller."""

from dataclasses import dataclass

import ai_arena_physics_base as physics


SPRING_ACTIONS = ("pelvis", "chest", "head", "left_leg", "right_leg")
ALL_SPRINGS = (1 << len(SPRING_ACTIONS)) - 1
POSE_ACTIONS = SPRING_ACTIONS + ("left_arm", "right_arm")
ALL_POSE_SPRINGS = (1 << len(POSE_ACTIONS)) - 1


@dataclass(frozen=True)
class SpringAction:
    mask: int


@dataclass(frozen=True)
class PoseAction:
    mask: int


def _apply_body_springs(points, mask, lift, align, damping,
                        pelvis_height, knee_offset, knee_drop, foot_offset):
    body = {point.name: point for point in points}
    pelvis = body["Pelvis"]
    chest = body["Chest"]
    head = body["Head"]

    if mask & 1:
        feet_middle = (body["Left Foot"].position.x
                       + body["Right Foot"].position.x) / 2
        physics.apply_target_spring(
            pelvis, (feet_middle, physics.FLOOR_Y - pelvis_height),
            lift, damping)
    if mask & 2:
        physics.apply_target_spring(
            chest, (pelvis.position.x, pelvis.position.y - 60),
            align, damping)
    if mask & 4:
        physics.apply_target_spring(
            head, (chest.position.x, chest.position.y - 50),
            align, damping)
    for bit, side, sign in ((8, "Left", -1), (16, "Right", 1)):
        if not mask & bit:
            continue
        physics.apply_target_spring(
            body[side + " Knee"],
            (pelvis.position.x + sign * knee_offset,
             pelvis.position.y + knee_drop),
            align, damping)
        physics.apply_target_spring(
            body[side + " Foot"],
            (pelvis.position.x + sign * foot_offset, physics.FLOOR_Y - 5),
            align, damping)
    return body


def apply_spring_action(points, mask, lift=160, align=80, damping=25):
    """Apply the original five feedback groups used by v6 models."""
    if not 0 <= mask <= ALL_SPRINGS:
        raise ValueError("Invalid spring action mask")
    _apply_body_springs(points, mask, lift, align, damping,
                        pelvis_height=140, knee_offset=25,
                        knee_drop=65, foot_offset=25)


def apply_pose_action(points, mask, lift=160, align=80, damping=25,
                      arm_strength=90, arm_damping=15):
    """Support a mirrored stance and hold each arm beside the torso."""
    if not 0 <= mask <= ALL_POSE_SPRINGS:
        raise ValueError("Invalid pose action mask")
    body = _apply_body_springs(
        points, mask, lift, align, damping,
        pelvis_height=135, knee_offset=34, knee_drop=58, foot_offset=30)
    chest = body["Chest"]
    for bit, side, sign in ((32, "Left", -1), (64, "Right", 1)):
        if not mask & bit:
            continue
        physics.apply_target_spring(
            body[side + " Elbow"],
            (chest.position.x + sign * 36, chest.position.y + 27),
            arm_strength, arm_damping)
        physics.apply_target_spring(
            body[side + " Hand"],
            (chest.position.x + sign * 42, chest.position.y + 72),
            arm_strength, arm_damping)
