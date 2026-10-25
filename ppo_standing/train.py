"""Train, resume, or evaluate an unassisted PPO standing policy."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
from xml.sax.saxutils import escape

import gymnasium
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import stable_baselines3
import torch

import ai_arena_physics_base as physics
from .policy import BoundedMeanPolicy
from .env import OBSERVATION_FIELDS, REWARD_FIELDS, PPOStandingEnv
from .env_v2 import PPOStableStanceEnv, REWARD_FIELDS_V2
from .env_v3 import PPOControlledStanceEnv, REWARD_FIELDS_V3
from .env_v4 import PPOCompactStanceEnv, REWARD_FIELDS_V4
from .env_v5 import PPOFiniteFrictionEnv, REWARD_FIELDS_V5


PACKAGE_SOURCE_FILES = frozenset((
    "env.py", "env_v2.py", "env_v3.py", "env_v4.py", "env_v5.py",
    "finite_friction.py", "policy.py", "train.py",
))
SOURCE_FILES = (
    "ai_arena_physics_base.py", "train_q_learning.py", "spring_control.py",
    "env.py", "train.py",
)
ENVIRONMENT_SOURCE_FILES = SOURCE_FILES[:-1]
EPISODE_METRICS = (
    "standing_seconds", "foot_contact_fraction", "valid_standing_fraction",
    "other_support_fraction", "wall_contact_fraction", "standing_success",
)


def code_hashes():
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    def source_path(name):
        return package_root / name if name in PACKAGE_SOURCE_FILES else project_root / name
    hashes = {name: hashlib.sha256(source_path(name).read_bytes()).hexdigest()
              for name in SOURCE_FILES}
    for name in ("env_v2.py", "policy.py", "env_v3.py", "env_v4.py",
                 "finite_friction.py", "env_v5.py"):
        hashes[name] = hashlib.sha256(source_path(name).read_bytes()).hexdigest()
    return hashes


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment_class(config):
    return {1: PPOStandingEnv, 2: PPOStableStanceEnv,
            3: PPOControlledStanceEnv,
            4: PPOCompactStanceEnv,
            5: PPOFiniteFrictionEnv}[config.get("env_version", 1)]


def environment_options(config):
    if config.get("env_version", 1) == 5:
        return {"static_mu": config.get("static_mu", 0.2),
                "kinetic_mu": config.get("kinetic_mu", 0.1)}
    return {}


def check_environment_hashes(manifest, current_hashes):
    required = list(ENVIRONMENT_SOURCE_FILES)
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
    if any(manifest["code_sha256"].get(name) != current_hashes[name]
           for name in required):
        raise ValueError("Physics/environment code changed since this run")


def evaluate_model(model, episode_seconds, first_seed, episodes,
                   env_version=1, env_options=None):
    """Evaluate on a fixed seed set without updating the policy."""
    env = environment_class({"env_version": env_version})(
        episode_seconds=episode_seconds, **(env_options or {}))
    reward_fields = {1: REWARD_FIELDS, 2: REWARD_FIELDS_V2,
                     3: REWARD_FIELDS_V3,
                     4: REWARD_FIELDS_V4,
                     5: REWARD_FIELDS_V5}[env_version]
    rows = []
    try:
        for seed in range(first_seed, first_seed + episodes):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            action_magnitude = []
            while True:
                action, _ = model.predict(observation, deterministic=True)
                action_magnitude.append(float(np.mean(np.abs(action))))
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            row = {
                "seed": seed,
                "reward": total_reward,
                "reached_time_limit": bool(truncated),
                "mean_action_magnitude": statistics.mean(action_magnitude),
                **{name: info[name] for name in EPISODE_METRICS},
                "reward_components": info["episode_reward_components"],
            }
            if env_version >= 2:
                row["stable_stance_fraction"] = info["stable_stance_fraction"]
                row["max_foot_span"] = info["max_foot_span"]
                row["terminal_foot_span"] = info["foot_span"]
            rows.append(row)
    finally:
        env.close()
    summary = {
        "episodes": episodes,
        "first_seed": first_seed,
        "median_standing_seconds": statistics.median(
            row["standing_seconds"] for row in rows),
        "mean_standing_seconds": statistics.mean(
            row["standing_seconds"] for row in rows),
        "success_rate": statistics.mean(
            row["standing_success"] for row in rows),
        "full_duration_rate": statistics.mean(
            row["reached_time_limit"] for row in rows),
        "mean_foot_contact_fraction": statistics.mean(
            row["foot_contact_fraction"] for row in rows),
        "mean_valid_standing_fraction": statistics.mean(
            row["valid_standing_fraction"] for row in rows),
        "mean_other_support_fraction": statistics.mean(
            row["other_support_fraction"] for row in rows),
        "mean_wall_contact_fraction": statistics.mean(
            row["wall_contact_fraction"] for row in rows),
        "mean_action_magnitude": statistics.mean(
            row["mean_action_magnitude"] for row in rows),
        "mean_reward": statistics.mean(row["reward"] for row in rows),
        "mean_reward_components": {
            name: statistics.mean(row["reward_components"][name] for row in rows)
            for name in reward_fields
        },
    }
    if env_version >= 2:
        summary["mean_stable_stance_fraction"] = statistics.mean(
            row["stable_stance_fraction"] for row in rows)
        summary["mean_max_foot_span"] = statistics.mean(
            row["max_foot_span"] for row in rows)
        summary["mean_terminal_foot_span"] = statistics.mean(
            row["terminal_foot_span"] for row in rows)
    return summary


def ranking(metrics):
    """Prefer real standing before training reward or action magnitude."""
    return (
        metrics["success_rate"],
        metrics["median_standing_seconds"],
        metrics["mean_valid_standing_fraction"],
        metrics.get("mean_stable_stance_fraction", 0),
        metrics["mean_foot_contact_fraction"],
        -metrics["mean_other_support_fraction"],
        -metrics["mean_wall_contact_fraction"],
    )


def write_progress_svg(run_dir):
    """Plot evaluation metrics without an extra plotting dependency."""
    rows = [json.loads(line) for line in
            (run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    episode_seconds = manifest["config"]["episode_seconds"]
    charts = (
        ("Standing time (seconds)", 0, episode_seconds,
         (("median", "median_standing_seconds", "#38bdf8"),
          ("mean", "mean_standing_seconds", "#f59e0b"))),
        ("Standing and contact fractions", 0, 1,
         (("success", "success_rate", "#22c55e"),
          ("valid pose", "mean_valid_standing_fraction", "#38bdf8"),
          ("foot contact", "mean_foot_contact_fraction", "#f59e0b"),
          ("other support", "mean_other_support_fraction", "#ef4444"),
          ("wall contact", "mean_wall_contact_fraction", "#a78bfa"))),
        ("Mean joint action magnitude", 0, 1,
         (("action", "mean_action_magnitude", "#38bdf8"),)),
    )
    palette = ("#38bdf8", "#f59e0b", "#22c55e", "#a78bfa", "#ef4444",
               "#fb7185", "#94a3b8", "#f97316", "#e879f9")
    recorded_reward_fields = manifest["reward_fields"]
    reward_fields = tuple(
        (name, name, palette[index % len(palette)])
        for index, name in enumerate(recorded_reward_fields))
    reward_values = [row["mean_reward_components"][name]
                     for row in rows for name in recorded_reward_fields]
    reward_min = min(0, min(reward_values))
    reward_max = max(0.1, max(reward_values))
    charts += (("Mean episode reward components", reward_min, reward_max,
                reward_fields),)
    width, height = 1000, 920
    x_left, x_right = 75, 965
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#0f172a"/>',
           '<style>text{font:13px sans-serif;fill:#e2e8f0}'
           '.title{font:bold 16px sans-serif}</style>',
           '<text x="75" y="30" class="title">PPO standing progress</text>']
    first_step, last_step = rows[0]["steps"], rows[-1]["steps"]
    for index, (title, low, high, series) in enumerate(charts):
        top = 58 + index * 215
        bottom = top + 145
        svg.append(f'<text x="75" y="{top}" class="title">{escape(title)}</text>')
        plot_top = top + 18
        for tick in range(4):
            y = plot_top + (bottom - plot_top) * tick / 3
            value = high - (high - low) * tick / 3
            svg.append(f'<line x1="{x_left}" y1="{y:.1f}" '
                       f'x2="{x_right}" y2="{y:.1f}" stroke="#334155"/>')
            svg.append(f'<text x="8" y="{y + 4:.1f}">{value:.2f}</text>')
        for label, field, color in series:
            samples = []
            for row in rows:
                x = (x_left + (x_right - x_left)
                     * (row["steps"] - first_step)
                     / max(last_step - first_step, 1))
                value = (row["mean_reward_components"][field]
                         if index == 3 else row[field])
                y = bottom - (bottom - plot_top) * (value - low) / (high - low)
                samples.append((x, y))
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in samples)
            svg.append(f'<polyline points="{points}" fill="none" '
                       f'stroke="{color}" stroke-width="2"/>')
            for x, y in samples:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" '
                           f'fill="{color}"/>')
        legend_y = bottom + 20
        legend_x = x_left
        for label, _, color in series:
            if legend_x + 27 + 8 * len(label) > x_right - 70:
                legend_x = x_left
                legend_y += 18
            svg.append(f'<rect x="{legend_x}" y="{legend_y - 9}" '
                       f'width="11" height="11" fill="{color}"/>')
            svg.append(f'<text x="{legend_x + 16}" y="{legend_y}">'
                       f'{escape(label)}</text>')
            legend_x += 27 + 8 * len(label)
        svg.append(f'<text x="{x_right - 100}" y="{plot_top + 15}">'
                   f'step {last_step}</text>')
    svg.append('</svg>')
    (run_dir / "progress.svg").write_text("\n".join(svg) + "\n",
                                           encoding="utf-8")


class StandingProgress(BaseCallback):
    def __init__(self, run_dir, manifest, eval_every, save_every):
        super().__init__()
        self.run_dir = run_dir
        self.manifest = manifest
        self.eval_every = eval_every
        self.save_every = save_every
        self.next_eval = (manifest.get("last_evaluation_step", 0)
                          + eval_every)
        self.next_save = (self.model_steps(manifest) // save_every + 1) * save_every

    @staticmethod
    def model_steps(manifest):
        return manifest.get("last_step", 0)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                append_jsonl(self.run_dir / "training_episodes.jsonl", {
                    "steps": self.num_timesteps,
                    "arena_seed": info["arena_seed"],
                    "reward": info["episode"]["r"],
                    **{name: info[name] for name in EPISODE_METRICS},
                    "reward_components": info["episode_reward_components"],
                    **({"stable_stance_fraction": info["stable_stance_fraction"],
                        "max_foot_span": info["max_foot_span"]}
                       if "stable_stance_fraction" in info else {}),
                })
        if self.num_timesteps >= self.next_eval:
            self.evaluate()
            self.next_eval = self.num_timesteps + self.eval_every
        if self.num_timesteps >= self.next_save:
            self.model.save(self.run_dir / "last.zip")
            self.manifest["last_step"] = self.num_timesteps
            write_json(self.run_dir / "manifest.json", self.manifest)
            self.next_save = self.num_timesteps + self.save_every
        return True

    def evaluate(self, phase="periodic"):
        config = self.manifest["config"]
        metrics = evaluate_model(
            self.model, config["episode_seconds"],
            config["eval_seed_start"], config["eval_episodes"],
            config.get("env_version", 1), environment_options(config))
        metrics["steps"] = self.num_timesteps
        metrics["phase"] = phase
        append_jsonl(self.run_dir / "evaluations.jsonl", metrics)
        write_progress_svg(self.run_dir)
        previous = self.manifest.get("best_evaluation")
        if previous is None or ranking(metrics) > ranking(previous):
            self.model.save(self.run_dir / "best.zip")
            self.manifest["best_evaluation"] = metrics
        self.manifest["last_evaluation_step"] = self.num_timesteps
        write_json(self.run_dir / "manifest.json", self.manifest)
        print(f"step={self.num_timesteps} "
              f"median={metrics['median_standing_seconds']:.3f}s "
              f"success={metrics['success_rate']:.0%} "
              f"valid={metrics['mean_valid_standing_fraction']:.2f}",
              flush=True)

    def _on_training_end(self):
        # _on_step evaluates before PPO trains on the most recent rollout.
        # Evaluate once more after the final optimizer update.
        self.evaluate(phase="final")
        self.model.save(self.run_dir / "last.zip")
        self.manifest["last_step"] = self.num_timesteps
        self.manifest["status"] = "finished"
        write_json(self.run_dir / "manifest.json", self.manifest)


def make_vector_env(config):
    def make_one():
        return Monitor(environment_class(config)(
            episode_seconds=config["episode_seconds"],
            **environment_options(config)))

    vector_env = DummyVecEnv([make_one for _ in range(config["num_envs"])])
    vector_env.seed(config["seed"])
    return vector_env


def new_manifest(config):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "last_step": 0,
        "config": config,
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pygame": pygame.version.ver,
        },
        "code_sha256": code_hashes(),
        "physics": {
            "fps": physics.FPS,
            "substeps": physics.SUBSTEPS,
            "constraint_iterations": physics.CONSTRAINT_ITERATIONS,
            "gravity_y": physics.GRAVITY_Y,
            "floor_y": physics.FLOOR_Y,
            "joint_actuators": physics.JOINT_ACTUATORS,
        },
        "observation_fields": OBSERVATION_FIELDS,
        "reward_fields": {1: REWARD_FIELDS, 2: REWARD_FIELDS_V2,
                          3: REWARD_FIELDS_V3,
                          4: REWARD_FIELDS_V4,
                          5: REWARD_FIELDS_V5}[config.get("env_version", 1)],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path, metavar="RUN_DIR")
    mode.add_argument("--evaluate", type=Path, metavar="MODEL_ZIP")
    mode.add_argument("--init-model", type=Path, metavar="MODEL_ZIP",
                      help="Start a new run from saved policy weights")
    parser.add_argument("--output-dir", type=Path, default=Path("ppo_runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=20_480,
                        help="Additional environment transitions")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--episode-seconds", type=float, default=3.0)
    parser.add_argument("--eval-episode-seconds", type=float,
                        help="Override episode length when using --evaluate")
    parser.add_argument("--env-version", type=int, choices=(1, 2, 3, 4, 5), default=1,
                        help="Task version: 1 original, 4 compact stance, 5 finite foot friction")
    parser.add_argument("--static-mu", type=float, default=0.2)
    parser.add_argument("--kinetic-mu", type=float, default=0.1)
    parser.add_argument("--policy", choices=("standard", "bounded"),
                        default="standard",
                        help="Action mean: standard clipped or bounded by tanh")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=2_048)
    parser.add_argument("--save-every", type=int, default=2_048)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.torch_threads < 1 or args.eval_episodes < 1:
        raise ValueError("Torch threads and evaluation episodes must be positive")
    torch.set_num_threads(args.torch_threads)
    if args.evaluate:
        path = args.evaluate.resolve()
        manifest_path = path.parent / "manifest.json"
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists() else None)
        current_hashes = code_hashes()
        if manifest:
            check_environment_hashes(manifest, current_hashes)
        episode_seconds = (manifest["config"]["episode_seconds"]
                           if manifest else args.episode_seconds)
        if args.eval_episode_seconds is not None:
            episode_seconds = args.eval_episode_seconds
        env_version = (manifest["config"].get("env_version", 1)
                       if manifest else args.env_version)
        model = PPO.load(path, device="cpu")
        print(json.dumps(evaluate_model(
            model, episode_seconds, args.eval_seed_start,
            args.eval_episodes, env_version,
            environment_options(manifest["config"] if manifest else {
                "env_version": env_version, "static_mu": args.static_mu,
                "kinetic_mu": args.kinetic_mu})),
            ensure_ascii=False, indent=2))
        return

    if args.timesteps < 1:
        raise ValueError("Timesteps must be positive")
    if args.resume:
        run_dir = args.resume.resolve()
        manifest = json.loads((run_dir / "manifest.json").read_text(
            encoding="utf-8"))
        current_hashes = code_hashes()
        check_environment_hashes(manifest, current_hashes)
        if manifest["code_sha256"]["train.py"] != current_hashes["train.py"]:
            manifest.setdefault("continuations", []).append({
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "training_script_sha256": current_hashes["train.py"],
            })
        config = manifest["config"]
        checkpoint = run_dir / "last.zip"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    else:
        if (args.seed < 0 or args.num_envs < 1 or args.n_steps < 2
                or args.batch_size < 2 or args.n_epochs < 1
                or args.eval_every < 1 or args.save_every < 1
                or args.eval_seed_start < 0):
            raise ValueError("Training counts and seeds must be positive")
        if (not 0 < args.learning_rate or not 0 < args.gamma <= 1
                or not 0 < args.gae_lambda <= 1 or args.ent_coef < 0
                or not 0 < args.clip_range):
            raise ValueError("Invalid PPO optimization settings")
        if (not np.isfinite(args.static_mu)
                or not np.isfinite(args.kinetic_mu)
                or args.static_mu < args.kinetic_mu
                or args.kinetic_mu < 0):
            raise ValueError("Require static_mu >= kinetic_mu >= 0")
        rollout_size = args.n_steps * args.num_envs
        if args.batch_size > rollout_size:
            raise ValueError("Batch size exceeds the PPO rollout size")
        if rollout_size <= 1:
            raise ValueError("PPO rollout needs at least two transitions")
        config = {
            "seed": args.seed, "num_envs": args.num_envs,
            "episode_seconds": args.episode_seconds,
            "env_version": args.env_version,
            "policy": args.policy,
            "n_steps": args.n_steps, "batch_size": args.batch_size,
            "n_epochs": args.n_epochs, "eval_every": args.eval_every,
            "save_every": args.save_every, "eval_episodes": args.eval_episodes,
            "eval_seed_start": args.eval_seed_start,
            "learning_rate": args.learning_rate, "gamma": args.gamma,
            "gae_lambda": args.gae_lambda, "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
        }
        if args.env_version == 5:
            config["static_mu"] = args.static_mu
            config["kinetic_mu"] = args.kinetic_mu
        name = (args.run_name or f"seed_{args.seed}_"
                + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        run_dir = (args.output_dir / name).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = new_manifest(config)
        if args.init_model:
            source_path = args.init_model.resolve()
            source_manifest_path = source_path.parent / "manifest.json"
            source_manifest = (json.loads(source_manifest_path.read_text(
                encoding="utf-8")) if source_manifest_path.exists() else None)
            current_hashes = code_hashes()
            if source_manifest:
                check_environment_hashes(source_manifest, current_hashes)
            manifest["initialization"] = {
                "model_path": str(source_path),
                "model_sha256": file_sha256(source_path),
                "source_episode_seconds": (
                    source_manifest["config"]["episode_seconds"]
                    if source_manifest else None),
                "method": "policy_weights_only_fresh_optimizer",
            }
        write_json(run_dir / "manifest.json", manifest)

    vector_env = make_vector_env(config)
    try:
        if args.resume:
            model = PPO.load(checkpoint, env=vector_env, device="cpu")
            print(f"Resuming {run_dir} from step {model.num_timesteps}", flush=True)
        else:
            model = PPO(
                (BoundedMeanPolicy if config.get("policy") == "bounded"
                 else "MlpPolicy"),
                vector_env, seed=config["seed"], device="cpu",
                n_steps=config["n_steps"], batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                learning_rate=config["learning_rate"], gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                clip_range=config["clip_range"], ent_coef=config["ent_coef"],
                verbose=0)
            if args.init_model:
                source = PPO.load(args.init_model.resolve(), device="cpu")
                if source.observation_space != model.observation_space:
                    raise ValueError("Source observation space does not match")
                if source.action_space != model.action_space:
                    raise ValueError("Source action space does not match")
                model.policy.load_state_dict(source.policy.state_dict())
                model.save(run_dir / "initial.zip")
            model.save(run_dir / "last.zip")
            print(f"Training {run_dir}", flush=True)
        manifest["status"] = "running"
        write_json(run_dir / "manifest.json", manifest)
        callback = StandingProgress(run_dir, manifest, config["eval_every"],
                                    config["save_every"])
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback,
                        reset_num_timesteps=not bool(args.resume),
                        progress_bar=False)
        except KeyboardInterrupt:
            model.save(run_dir / "last.zip")
            manifest["last_step"] = model.num_timesteps
            manifest["status"] = "interrupted"
            write_json(run_dir / "manifest.json", manifest)
            print(f"Interrupted at step {model.num_timesteps}", flush=True)
    finally:
        vector_env.close()


if __name__ == "__main__":
    main()
