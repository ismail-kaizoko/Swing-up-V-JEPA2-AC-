"""
Pure dynamics of a torque-limited simple pendulum.

State:   theta      - angle from upright, radians. theta=0 is the unstable
                       equilibrium (goal). theta=+-pi is hanging straight
                       down (stable equilibrium, where episodes start).
         theta_dot  - angular velocity (rad/s)

Action:  u in [-u_max, u_max] - torque applied at the pivot (N*m)

Equation of motion (torque balance about the pivot, point mass m at the
rod tip so I = m*l^2):

    I * theta_ddot = m*g*l*sin(theta) - b*theta_dot + u

Dividing by I:

    theta_ddot = (g/l)*sin(theta) - (b/(m*l^2))*theta_dot + u/(m*l^2)

The sin(theta) term is the whole point of this project: it is
*destabilizing* at theta=0 (gravity pulls the pendulum away from
upright) and *restoring* at theta=+-pi (gravity pulls it back down).
Nothing in this equation offers a shortcut — reaching theta=0 from
theta=pi under a torque limit requires trading potential/kinetic
energy back and forth across multiple swings.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class PendulumParams:
    m: float = 1.0          # ball mass (kg)
    l: float = 1.0          # rod length (m)
    g: float = 9.81         # gravity (m/s^2)
    b: float = 0.2            # damping / friction coefficient
    dt: float = 0.01        # integration timestep (s), i.e. 100 Hz control
    max_speed: float = 8.0  # clip |theta_dot| to avoid numerical blow-up
    T_max = m*g*l/10        # torque limit (N*m) — deliberately << m*g*l

def step(theta: np.ndarray, theta_dot: np.ndarray, u: np.ndarray, p: PendulumParams):
    """
    Advance one timestep via semi-implicit (symplectic) Euler:
        v_{t+1} = v_t + a_t * dt
        x_{t+1} = x_t + v_{t+1} * dt          <- uses the *new* velocity
    Ordinary (explicit) Euler uses the *old* velocity for the position
    update, which slowly injects energy into oscillatory systems — your
    pendulum would drift toward larger and larger swings for free, which
    is a purely numerical artifact, not a real strategy. Semi-implicit
    Euler is what every physics engine and gym's Pendulum-v1 use instead,
    because it's symplectic: it conserves energy on average.
    """
    T = np.clip(u, -p.T_max, p.T_max)

    theta_ddot = (
        (p.g / p.l) * np.sin(theta)
        - (p.b / (p.m * p.l ** 2)) * theta_dot
        + T / (p.m * p.l ** 2)
    )

    # theta_dot_new = np.clip(theta_dot + theta_ddot * p.dt, -p.max_speed, p.max_speed)
    theta_dot_new = theta_dot + theta_ddot * p.dt
    theta_new = theta + theta_dot_new * p.dt

    # wrap into (-pi, pi]: atan2(sin(x), cos(x)) is the standard trick for
    # angle-wrapping — it maps any real x back onto the unit circle without
    # the branch-boundary bugs a manual `% (2*pi)` tends to introduce.
    # theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

    return theta_new, theta_dot_new