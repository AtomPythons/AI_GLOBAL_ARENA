"""Watch a saved Q-learning policy control the humanoid."""

import argparse
from pathlib import Path

import pygame

import ai_arena_physics_base as physics
from train_q_learning import QAgent, STATUS_HEIGHT, run_demo


def latest_model():
    models = sorted(Path(__file__).parent.glob("ai_model_*.json"))
    if not models:
        raise SystemExit("No model found. Train one first or pass --model PATH.")
    return models[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Saved ai_model_*.json file")
    args = parser.parse_args()
    path = args.model or latest_model()
    agent = QAgent.load(path)
    pygame.init()
    try:
        screen = pygame.display.set_mode(
            (physics.SCREEN_WIDTH, physics.SCREEN_HEIGHT + STATUS_HEIGHT))
        pygame.display.set_caption("AI Arena - Saved model")
        font = pygame.font.SysFont(None, 20)
        small_font = pygame.font.SysFont(None, 16)
        run_demo(screen, font, small_font, agent, path.name)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
