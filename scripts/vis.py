"""
Rollout visualizer. Two outputs per episode:
  1. An actual video (.mp4) of the rendered frames — watch the pendulum
     swing instead of squinting at arrays.
  2. A static plot of theta(t) and T(t) — the fast quantitative check
     from earlier (does swing amplitude grow across the episode) without
     opening every clip by hand.
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from pathlib import Path


def load_episode(path):
    data = np.load(path)
    return data["frames"], data["actions"], data["states"]


def save_video(frames: np.ndarray, out_path: Path, fps: int = 20):
    # frames: (T, H, W) uint8 grayscale -> most video codecs expect 3
    # channels, so broadcast grayscale to RGB by repeating the channel.
    rgb_frames = np.repeat(frames[..., None], 3, axis=-1)
    iio.imwrite(out_path, rgb_frames, fps=fps, codec="libx264")


def plot_trajectory(actions: np.ndarray, states: np.ndarray, dt: float, out_path: Path):
    t = np.arange(len(states)) * dt
    theta, theta_dot = states[:, 0], states[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(t, theta, label="theta")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1, label="upright (goal)")
    axes[0].set_ylabel("theta (rad)")
    axes[0].legend(loc="upper right")

    axes[1].plot(t[:-1], actions, color="tab:orange")
    axes[1].set_ylabel("T (torque)")
    axes[1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def visualize_episode(npz_path: str, out_dir: str = "data/visualizations", dt: float = 0.05):
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, actions, states = load_episode(npz_path)
    stem = npz_path.stem

    save_video(frames, out_dir / f"{stem}.mp4")
    plot_trajectory(actions, states, dt, out_dir / f"{stem}_traj.png")
    print(f"saved {out_dir / (stem + '.mp4')} and {out_dir / (stem + '_traj.png')}")


if __name__ == "__main__":  
    import sys
    visualize_episode(sys.argv[1])