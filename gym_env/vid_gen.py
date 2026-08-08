import gymnasium as gym
import numpy as np


scenarios = [
    {"theta": 0.0, "theta_dot": 0.0},
    {"theta": np.pi / 4, "theta_dot": 0.0},
    {"theta": np.pi / 2, "theta_dot": 0.0},
    {"theta": np.pi, "theta_dot": 0.0},
    {"theta": -np.pi / 2, "theta_dot": 0.0},
]

env = gym.make(
    "Pendulum-v1",
    render_mode="rgb_array")


env = gym.wrappers.RecordVideo(
    env,
    video_folder="./data/gym_videos",
    episode_trigger=lambda episode: True,
)

for scenario in scenarios:

    obs, info = env.reset(
        options=scenario
    )

    for _ in range(10000):

        action = np.array([0.0])

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break