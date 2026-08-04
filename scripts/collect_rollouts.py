"""
Random-rollout collector — the data-generation stage JEPA-1-AC pretrains
on. No reward, no policy: just diverse (frame_t, action_t, frame_{t+1})
transitions covering the state-action space, so what gets learned is the
*dynamics*, not any one trajectory. CEM planning (stage 3) is what later
searches this learned model for the swing-up strategy.

Why not i.i.d. random torque? At 20 Hz, independent random torque per step
mostly cancels out over a few steps (its net effect on velocity averages
toward zero) — the pendulum barely leaves its starting angle. Instead we
sample torque from an Ornstein-Uhlenbeck (OU) process: a temporally
correlated "mean-reverting random walk," the standard exploration-noise
trick from DDPG. It naturally produces smooth push-pull torque sequences,
which is exactly the kind of action pattern needed to pump energy into
the pendulum and see a wide range of swing amplitudes in the dataset.

OU discretized update:
    u_{t+1} = u_t + theta_ou*(mu - u_t)*dt + sigma*sqrt(dt)*eps_t,  eps_t ~ N(0,1)
"""
import numpy as np
from pathlib import Path
from env.physics import PendulumParams
from env.pendulum_env import PendulumSwingUpEnv


def ou_action_sequence(rng, n_steps, dt, u_max, theta_ou=1.0, sigma=3.0):
    u = np.zeros(n_steps, dtype=np.float32)
    for t in range(1, n_steps):
        eps = rng.standard_normal()
        u[t] = u[t - 1] + theta_ou * (0.0 - u[t - 1]) * dt + sigma * np.sqrt(dt) * eps
    return np.clip(u, -u_max, u_max)


def collect_episode(env, rng, n_steps):
    frames = np.zeros((n_steps + 1, env.renderer.img_size, env.renderer.img_size), dtype=np.uint8)
    actions = np.zeros(n_steps, dtype=np.float32)
    states = np.zeros((n_steps + 1, 2), dtype=np.float32)  # (theta, theta_dot) — eval-only

    frames[0] = env.reset(rng)
    states[0] = (env.theta, env.theta_dot)

    u_seq = ou_action_sequence(rng, n_steps, env.p.dt, env.p.u_max)
    for t in range(n_steps):
        obs, info = env.step(u_seq[t])
        frames[t + 1] = obs
        actions[t] = u_seq[t]
        states[t + 1] = (info["theta"], info["theta_dot"])

    return frames, actions, states


def main(n_episodes: int = 2000, n_steps: int = 100, img_size: int = 64,
         out_dir: str = "data/rollouts", seed: int = 0):
    rng = np.random.default_rng(seed)
    env = PendulumSwingUpEnv(img_size=img_size)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for ep in range(n_episodes):
        frames, actions, states = collect_episode(env, rng, n_steps)
        np.savez_compressed(out_path / f"episode_{ep:05d}.npz",
                             frames=frames, actions=actions, states=states)
        if (ep + 1) % 200 == 0:
            print(f"collected {ep + 1}/{n_episodes} episodes")


if __name__ == "__main__":
    main()