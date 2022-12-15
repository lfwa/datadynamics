from pettingzoo.utils.env import AECEnv
import pygame
import numpy as np
import gymnasium

FPS = 120
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000


def env(**kwargs):
    env = raw_env(**kwargs)
    return env


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "is_parrallelizable": False,
        "render_fps": FPS,
    }

    def __init__(self, render_mode=None) -> None:
        assert (
            render_mode in self.metadata["render_modes"] or render_mode is None
        ), f"render_mode: {render_mode} is not supported. Supported modes: {self.metadata['render_modes']}"

        self.seed()

        self.render_mode = render_mode

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed)

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.rng, seed = gymnasium.utils.seeding.np_random(seed)
        return [seed]
