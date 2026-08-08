import gymnasium as gym

env = gym.make(
    "Pendulum-v1",
    render_mode="rgb_array",
)

env = gym.wrappers.RecordVideo(
    env,
    video_folder="./data/gym_videos",
    episode_trigger=lambda episode: True,
)

obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()