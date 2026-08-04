"""
Minimal gym-like wrapper: holds the physics state, advances it via
physics.step, and returns an *image* observation, not (theta, theta_dot).
JEPA only ever sees pixels — the true state is kept in `info` purely for
your own evaluation/debugging, never fed to the model.
"""
import numpy as np
from env.physics import PendulumParams, step
from env.renderer import PendulumRenderer


class PendulumSwingUpEnv:
    def __init__(self, params: PendulumParams = None, img_size: int = 64):
        self.p = params or PendulumParams()
        self.renderer = PendulumRenderer(img_size=img_size)
        self.theta = None
        self.theta_dot = None

    def reset(self, rng: np.random.Generator, noise: float = 0.1):
        # start hanging down (theta=pi) with a small random perturbation
        # so rollouts within a dataset aren't all identical
        self.theta = np.pi + rng.uniform(-noise, noise)
        self.theta_dot = rng.uniform(-noise, noise)
        return self._obs()

    def step(self, u: float):
        theta, theta_dot = step(
            np.array(self.theta), np.array(self.theta_dot), np.array(u), self.p
        )
        self.theta, self.theta_dot = float(theta), float(theta_dot)
        return self._obs(), {"theta": self.theta, "theta_dot": self.theta_dot}

    def _obs(self):
        return self.renderer.render(self.theta)