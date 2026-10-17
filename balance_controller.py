"""Compare standing controllers and inspect their poses."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from statistics import mean, median

import pygame

import ai_arena_physics_base as physics
from spring_control import (ALL_POSE_SPRINGS, ALL_SPRINGS,
                            apply_pose_action, apply_spring_action)
from train_q_learning import (ACTION_FRAMES, Arena, QAgent, STATUS_HEIGHT,
                              draw_arena, encode_state)


class BalanceController:
    """Spring forces around a standing pose and the current foot support."""

    def __init__(self, lift=160, align=80, damping=25):
        self.lift = lift
        self.align = align
        self.damping = damping

    def apply(self, points):
        apply_spring_action(points, ALL_SPRINGS,
                            self.lift, self.align, self.damping)


class PoseController:
    """A mirrored standing pose with active arms for inspection."""

    def apply(self, points):
        apply_pose_action(points, ALL_POSE_SPRINGS)


class DiscreteBalanceController:
    """Feedback using only the eleven actions available to Q-learning."""

    def __init__(self):
        self.action_by_name = {name: index for index, name
                               in enumerate(physics.get_action_space())}

    def greedy_action(self, state):
        lean, offset = state[:2]
        if lean >= 4:
            return self.action_by_name["chest_right"]
        if lean <= 2:
            return self.action_by_name["chest_left"]
        if offset >= 3:
            return self.action_by_name["pelvis_left"]
        if offset <= 1:
            return self.action_by_name["pelvis_right"]
        return self.action_by_name["chest_up"]

    def decode_action(self, action):
        return action


@dataclass(frozen=True)
class EpisodeResult:
    alive: float
    contact_fraction: float
    mean_standing_score: float


def run_episode(seed, controller=None, model=None):
    arena = Arena()
    arena.reset(seed)
    frames_left = 0
    action = 0
    frames = 0
    contact_frames = 0
    standing_scores = 0
    while not arena.episode.is_episode_over:
        if frames_left == 0:
            state = (model.encode_state(arena.points, arena.metrics)
                     if hasattr(model, "encode_state")
                     else encode_state(arena.points, arena.metrics))
            action = (model.greedy_action(state)
                      if model else 0)
            frames_left = ACTION_FRAMES
        arena.step(model.decode_action(action) if model else 0,
                   controller=controller)
        frames_left -= 1
        frames += 1
        contact_frames += (arena.metrics["left_foot_contact"]
                           or arena.metrics["right_foot_contact"])
        standing_scores += arena.metrics["standing_score"]
    return EpisodeResult(arena.episode.time_alive,
                         contact_frames / frames,
                         standing_scores / frames)


def summarize(name, results):
    times = [result.alive for result in results]
    print(f"{name:18} mean {mean(times):.3f}s  median {median(times):.3f}s  "
          f">=1s {sum(t >= 1 for t in times)}/{len(times)}  "
          f">=3s {sum(t >= 3 for t in times)}/{len(times)}  "
          f"foot contact {mean(r.contact_fraction for r in results):.1%}  "
          f"score {mean(r.mean_standing_score for r in results):.2f}/5")


def evaluate(episodes, controller, model=None, controller_name="Balance controller"):
    seeds = range(1000, 1000 + episodes)
    summarize("No control", [run_episode(seed) for seed in seeds])
    discrete = DiscreteBalanceController()
    summarize("Discrete feedback",
              [run_episode(seed, model=discrete) for seed in seeds])
    summarize(controller_name,
              [run_episode(seed, controller=controller) for seed in seeds])
    if model is not None:
        summarize("Saved Q-model",
                  [run_episode(seed, model=model) for seed in seeds])


def show(controller):
    pygame.init()
    try:
        screen = pygame.display.set_mode(
            (physics.SCREEN_WIDTH, physics.SCREEN_HEIGHT + STATUS_HEIGHT))
        pygame.display.set_caption("AI Arena - Balance controller")
        font = pygame.font.SysFont(None, 20)
        small_font = pygame.font.SysFont(None, 16)
        clock = pygame.time.Clock()
        arena = Arena(episode_seconds=10)
        selected = arena.points[0]
        show_labels = False
        enabled = True
        dragging = False
        number_keys = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                       pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
                       pygame.K_9, pygame.K_0, pygame.K_MINUS)
        running = True
        while running:
            clock.tick(physics.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        arena.reset()
                        selected = arena.points[0]
                        dragging = False
                    elif event.key == pygame.K_n:
                        arena.episode.next_episode()
                        arena.reset(random.randrange(1_000_000))
                        selected = arena.points[0]
                        dragging = False
                    elif event.key == pygame.K_m:
                        enabled = not enabled
                    elif event.key == pygame.K_p:
                        arena.posture_assist = not arena.posture_assist
                    elif event.key == pygame.K_l:
                        show_labels = not show_labels
                    elif event.key in number_keys:
                        selected = arena.points[number_keys.index(event.key)]
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked = physics.find_point_at_mouse(arena.points, event.pos)
                    if clicked is not None:
                        selected = clicked
                        dragging = True
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False
            if not arena.episode.is_episode_over:
                arena.step(0, selected, pygame.key.get_pressed(),
                           pygame.mouse.get_pos() if dragging else None,
                           controller if enabled else None)
            lines = [f"Controller: {'on' if enabled else 'off'}  "
                     f"Standing time: {arena.episode.time_alive:.2f}s",
                     "R replay  N new pose  M controller  P assist",
                     "L labels  1-0/- select  Arrows/mouse move point"]
            draw_arena(screen, font, small_font, arena, selected,
                       show_labels, lines)
    finally:
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of fixed test poses")
    parser.add_argument("--model", type=Path, help="Q-model to compare")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive demonstration")
    parser.add_argument("--pose", action="store_true",
                        help="Use the arm-and-leg pose controller")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("Episode count must be positive")
    controller = PoseController() if args.pose else BalanceController()
    model = QAgent.load(args.model) if args.model else None
    name = "Pose controller" if args.pose else "Balance controller"
    evaluate(args.episodes, controller, model, name)
    if args.show:
        show(controller)


if __name__ == "__main__":
    main()
