"""Watch a saved PPO standing policy in the same physics environment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO

import ai_arena_physics_base as physics
from .env import PPOStandingEnv
from .env_v2 import PPOStableStanceEnv
from .env_v3 import PPOControlledStanceEnv
from .env_v4 import PPOCompactStanceEnv
from .env_v5 import PPOFiniteFrictionEnv


PANEL_WIDTH = 360
WINDOW_SIZE = (physics.SCREEN_WIDTH + PANEL_WIDTH, physics.SCREEN_HEIGHT)
PANEL_BACKGROUND = (14, 21, 32)
MUTED_COLOR = (160, 174, 192)
GOOD_COLOR = (95, 220, 155)
BAD_COLOR = (255, 120, 110)
ACTION_NAMES = tuple(item[0] for item in physics.JOINT_ACTUATORS)


def best_available_model():
    model_dir = Path(__file__).resolve().parent.parent / "ppo_runs"
    models = list(model_dir.glob("*/best.zip"))
    if not models:
        raise FileNotFoundError(
            "No PPO best.zip found. Pass --model or train a PPO model first.")
    completed = []
    for path in models:
        manifest_path = path.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("last_step", 0) >= 1024:
                completed.append(path)
    if completed:
        models = completed

    def score(path):
        manifest_path = path.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = manifest.get("best_evaluation")
            if metrics:
                return (1, metrics["median_standing_seconds"],
                        metrics["success_rate"],
                        metrics["mean_valid_standing_fraction"],
                        metrics["mean_foot_contact_fraction"],
                        -metrics["mean_other_support_fraction"],
                        -metrics["mean_wall_contact_fraction"])
        return (0, path.stat().st_mtime)

    return max(models, key=score)


def load_model_environment(path, episode_seconds=None):
    path = Path(path).resolve()
    manifest_path = path.parent / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else None)
    if manifest:
        source_hashes = manifest.get("code_sha256", {})
        required = ["ai_arena_physics_base.py", "train_q_learning.py",
                    "spring_control.py", "env.py"]
        if manifest["config"].get("env_version", 1) >= 2:
            required.append("env_v2.py")
        if manifest["config"].get("env_version", 1) >= 3:
            required.append("env_v3.py")
        if manifest["config"].get("env_version", 1) >= 4:
            required.append("env_v4.py")
        if manifest["config"].get("env_version", 1) == 5:
            required.extend(("finite_friction.py", "env_v5.py"))
        if manifest["config"].get("policy", "standard") == "bounded":
            required.append("policy.py")
        for name in required:
            expected = source_hashes.get(name)
            package_root = Path(__file__).resolve().parent
            project_root = package_root.parent
            package_files = {"env.py", "env_v2.py", "env_v3.py", "env_v4.py",
                             "env_v5.py", "finite_friction.py", "policy.py"}
            actual = hashlib.sha256(
                ((package_root if name in package_files else project_root) / name)
                .read_bytes()).hexdigest()
            if expected is not None and actual != expected:
                raise ValueError(f"Physics/environment code changed: {name}")
    duration = (episode_seconds if episode_seconds is not None else
                manifest["config"]["episode_seconds"] if manifest else 3.0)
    version = manifest["config"].get("env_version", 1) if manifest else 1
    env_class = {1: PPOStandingEnv, 2: PPOStableStanceEnv,
                 3: PPOControlledStanceEnv,
                 4: PPOCompactStanceEnv,
                 5: PPOFiniteFrictionEnv}[version]
    options = ({"static_mu": manifest["config"].get("static_mu", 0.2),
                "kinetic_mu": manifest["config"].get("kinetic_mu", 0.1)}
               if version == 5 else {})
    env = env_class(episode_seconds=duration, **options)
    try:
        model = PPO.load(path, device="cpu")
        if model.observation_space != env.observation_space:
            raise ValueError("Model observation space does not match this environment")
        if model.action_space != env.action_space:
            raise ValueError("Model action space does not match this environment")
    except Exception:
        env.close()
        raise
    return model, env


def draw_view(screen, font, small_font, env, info, action, model_enabled,
              paused, finished, show_labels, seed, model_name, total_reward,
              calibrated=False):
    """Draw the current physical state and a readable diagnostics panel."""
    screen.fill(physics.BACKGROUND)
    pygame.draw.rect(screen, physics.FLOOR_COLOR,
                     (0, physics.FLOOR_Y, physics.SCREEN_WIDTH,
                      physics.SCREEN_HEIGHT - physics.FLOOR_Y))
    pygame.draw.line(screen, physics.BONE_COLOR,
                     (0, physics.FLOOR_Y),
                     (physics.SCREEN_WIDTH, physics.FLOOR_Y), 2)
    for bone in env.arena.bones:
        bone.draw(screen)
    for point in env.arena.points:
        point.draw(screen)
        if show_labels:
            label = small_font.render(point.name, True, physics.TEXT_COLOR)
            screen.blit(label, point.position + (8, -12))

    panel_x = physics.SCREEN_WIDTH
    pygame.draw.rect(screen, PANEL_BACKGROUND,
                     (panel_x, 0, PANEL_WIDTH, physics.SCREEN_HEIGHT))
    x = panel_x + 16

    def line(label, y, color=physics.TEXT_COLOR):
        screen.blit(font.render(label, True, color), (x, y))

    line("PPO standing", 14)
    line(model_name[:38], 38, MUTED_COLOR)
    line(f"Seed: {seed}   Assist: OFF", 70, GOOD_COLOR)
    state = ("FELL" if info["is_fallen"] else
             "SUCCESS" if finished and info["standing_success"] else
             "FINISHED" if finished else
             "PAUSED" if paused else "STANDING")
    line(f"State: {state}", 94,
         BAD_COLOR if info["is_fallen"] else GOOD_COLOR)
    line(f"Time: {info['elapsed_seconds']:.2f} / "
         f"{env.max_frames / physics.FPS:.2f} s", 120)
    line(f"Standing: {info['standing_seconds']:.2f} s", 144)
    line(f"Pelvis height: {info['pelvis_height']:.1f}", 168)
    line(f"Torso angle: {info['torso_angle']:.1f} deg", 192)
    contacts = env.arena.metrics
    line("Feet: " +
         f"L {'yes' if contacts['left_foot_contact'] else 'no'}  "
         f"R {'yes' if contacts['right_foot_contact'] else 'no'}", 216)
    line(f"Valid pose: {info['valid_standing_fraction']:.0%}  "
         f"Other support: {info['other_support_fraction']:.0%}", 240)
    body = {point.name: point for point in env.arena.points}
    span = abs(body["Right Foot"].position.x - body["Left Foot"].position.x)
    stance = (f"{info['stable_stance_fraction']:.0%}"
              if "stable_stance_fraction" in info else "n/a")
    line(f"Foot span: {span:.0f}  Stable stance: {stance}", 264)
    line(f"Wall contact: {info['wall_contact_fraction']:.0%}  "
         f"Reward: {total_reward:.2f}", 288)
    line("Joint commands", 312, MUTED_COLOR)
    for index, (name, value) in enumerate(zip(ACTION_NAMES, action)):
        y = 335 + index * 22
        line(f"{name:>11}  {value:+.2f}", y)
        bar_x = x + 218
        pygame.draw.rect(screen, (54, 67, 83), (bar_x, y + 4, 105, 9))
        middle = bar_x + 52
        pygame.draw.line(screen, MUTED_COLOR,
                         (middle, y + 2), (middle, y + 15))
        extent = round(float(value) * 50)
        if extent:
            pygame.draw.rect(screen,
                             GOOD_COLOR if extent > 0 else BAD_COLOR,
                             (min(middle, middle + extent), y + 5,
                              abs(extent), 7))
    pygame.draw.line(screen, (54, 67, 83), (x, 452),
                     (panel_x + PANEL_WIDTH - 16, 452))
    mode = ("PPO + audit correction" if calibrated else "PPO") if model_enabled \
        else "zero action"
    line(f"Mode: {mode}", 459)
    line("R replay   N next   M compare", 477, MUTED_COLOR)
    # Space pauses, L toggles labels, Esc quits (also shown in window caption).
    pygame.display.flip()


def watch(model, env, model_name, first_seed, corrections=None):
    corrections = corrections or {}
    pygame.init()
    try:
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(
            "AI Arena PPO | R replay | N next | M compare | Space pause | L labels | Esc quit")
        font = pygame.font.SysFont(None, 20)
        small_font = pygame.font.SysFont(None, 16)
        clock = pygame.time.Clock()
        seed = first_seed
        observation, info = env.reset(seed=seed)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        total_reward = 0.0
        finished = False
        paused = False
        model_enabled = True
        show_labels = False
        running = True
        while running:
            clock.tick(physics.FPS / env.control_frames)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        observation, info = env.reset(seed=seed)
                        action.fill(0)
                        total_reward = 0.0
                        finished = False
                        paused = False
                    elif event.key == pygame.K_n:
                        seed += 1
                        observation, info = env.reset(seed=seed)
                        action.fill(0)
                        total_reward = 0.0
                        finished = False
                        paused = False
                    elif event.key == pygame.K_m:
                        model_enabled = not model_enabled
                        observation, info = env.reset(seed=seed)
                        action.fill(0)
                        total_reward = 0.0
                        finished = False
                        paused = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_l:
                        show_labels = not show_labels
            if not running:
                break
            if not paused and not finished:
                if model_enabled:
                    action = model.predict(observation, deterministic=True)[0]
                    if seed in corrections:
                        scale, bias = corrections[seed]
                        action = np.clip(action * scale + bias, -1, 1).astype(
                            np.float32)
                else:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                finished = terminated or truncated
            draw_view(screen, font, small_font, env, info, action,
                      model_enabled, paused, finished, show_labels,
                      seed, model_name, total_reward,
                      calibrated=seed in corrections)
    finally:
        pygame.quit()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        help="PPO .zip model (default: best evaluated run)")
    parser.add_argument("--seed", type=int, default=30_000,
                        help="Starting pose seed (default: 30000)")
    parser.add_argument("--episode-seconds", type=float,
                        help="Override the model run's episode duration")
    parser.add_argument("--audit-result", type=Path,
                        help="Use saved per-seed joint corrections")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be non-negative")
    corrections = {}
    duration = args.episode_seconds
    if args.audit_result:
        audit = json.loads(args.audit_result.read_text(encoding="utf-8"))
        path = args.model or Path(audit["config"]["model"])
        expected = audit["config"]["model_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error("Audit corrections belong to a different model")
        duration = duration or audit["config"]["seconds"]
        for row in audit["episodes"]:
            best = row["best"]
            scale = np.asarray(best["scale"], dtype=np.float32)
            bias = np.asarray(best["bias"], dtype=np.float32)
            if (scale.shape != (len(ACTION_NAMES),)
                    or bias.shape != scale.shape
                    or not np.isfinite(scale).all()
                    or not np.isfinite(bias).all()):
                parser.error("Invalid audit correction")
            corrections[row["seed"]] = (scale, bias)
        if args.seed not in corrections:
            parser.error("No audit correction for the requested seed")
    else:
        path = args.model or best_available_model()
    model, env = load_model_environment(path, duration)
    watch(model, env, f"{path.parent.name}/{path.name}", args.seed,
          corrections)


if __name__ == "__main__":
    main()
