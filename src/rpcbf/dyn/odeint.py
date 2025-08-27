import jax.lax as lax
import numpy as np
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from og.dyn_types import State
from og.jax_types import FloatScalar


def rk4(dt: FloatScalar, xdot, x1: State) -> State:
    k1 = xdot(x1)

    x2 = x1 + k1 * dt * 0.5
    k2 = xdot(x2)

    x3 = x1 + k2 * dt * 0.5
    k3 = xdot(x3)

    x4 = x1 + k3 * dt
    k4 = xdot(x4)

    return x1 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk4_multi(dt: FloatScalar, xdot, x1: State, n_steps: int) -> State:
    def body(x, _):
        x_new = rk4(dt_substep, xdot, x)
        return x_new, None

    dt_substep = dt / n_steps
    x_out, _ = lax.scan(body, x1, None, length=n_steps)
    return x_out
