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
    T_{t+1} = T_t + theta_ou*(mu - u_t)*dt + sigma*sqrt(dt)*eps_t,  eps_t ~ N(0,1)
"""
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from itertools import product
from pathlib import Path
from env.physics import PendulumParams
from env.pendulum_env import PendulumSwingUpEnv





def ou_action_sequence(rng, n_steps, dt, T_max, theta_oT=1.0, sigma=3.0):
    T = np.zeros(n_steps, dtype=np.float32)
    for t in range(1, n_steps):
        eps = rng.standard_normal()
        T[t] = T[t - 1] + theta_oT * (0.0 - T[t - 1]) * dt + sigma * np.sqrt(dt) * eps
    return np.clip(T, -T_max, T_max)


def collect_episode(env, rng, n_steps, free = True):
    "collects episodes "
    "free : when True, it releases the pendulum with 0 tork during the whole episode. "
    frames = np.zeros((n_steps + 1, env.renderer.img_size, env.renderer.img_size), dtype=np.uint8)
    actions = np.zeros(n_steps, dtype=np.float32)
    states = np.zeros((n_steps + 1, 2), dtype=np.float32)  # (theta, theta_dot) — eval-only

    frames[0] = env.reset(rng)
    states[0] = (env.theta, env.theta_dot)

    if free : 
        T_seq = np.zeros(n_steps)
    elif  free == "ou_process" :
        T_seq = ou_action_sequence(rng, n_steps, env.p.dt, env.p.T_max)
    else : assert("specify valid sequence generation")

    for t in range(n_steps):
        obs, info = env.step(T_seq[t])
        frames[t + 1] = obs
        actions[t] = T_seq[t]
        states[t + 1] = (info["theta"], info["theta_dot"])

    return frames, actions, states


def main(n_episodes: int = 100, duration: int = 20, img_size: int = 64,
         out_dir: str = "data/rollouts/test", seed: int = 0):

    rng = np.random.default_rng(seed)
    env = PendulumSwingUpEnv(img_size=img_size)
    n_frames = int(duration / env.p.dt)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    theta0 = np.pi/4
    theta_dot0 = 0
    scenario=0

    env.theta, env.theta_dot = float(theta0), float(theta_dot0)
    frames, actions, states = collect_episode(env, rng, n_frames)
    np.savez_compressed(out_path / f"episode_{scenario:05d}.npz",
                            frames=frames, actions=actions, states=states)



if __name__ == "__main__":
    main()